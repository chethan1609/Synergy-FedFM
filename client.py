import numpy as np
import torch
from typing import Dict, Tuple
from model import SynergyModel

class WelfordScaler:
    def __init__(self, num_features: int):
        self.n = 0.0
        self.mean = np.zeros(num_features, dtype=np.float64)
        self.M2 = np.zeros(num_features, dtype=np.float64)
        self.is_frozen = False

    def partial_fit(self, batch: np.ndarray):
        if self.is_frozen or batch.shape[0] == 0: return
        n_b = float(batch.shape[0])
        mean_b = batch.mean(0)
        m2_b = ((batch - mean_b) ** 2).sum(0)
        delta = mean_b - self.mean
        self.n += n_b
        self.mean += delta * (n_b / self.n)
        self.M2 += m2_b + (delta ** 2) * ((self.n - n_b) * n_b / self.n)

    def transform(self, batch: np.ndarray):
        if self.n < 2: return batch.astype(np.float32)
        std = np.sqrt(self.M2 / (self.n - 1) + 1e-8)
        return ((batch - self.mean) / std).astype(np.float32)

    def freeze(self):
        self.is_frozen = True

class SFCFilter:
    """SFC with JL projection + novelty threshold."""
    def __init__(self, input_dim: int, jl_dim: int, max_memory: int, base_gamma: float):
        self.max_memory  = max_memory
        self.base_gamma  = base_gamma
        self.kappa       = 2.0
        self.beta        = 0.1
        rng = np.random.default_rng(seed=42)
        self.jl_matrix = (rng.standard_normal((input_dim, jl_dim)) / np.sqrt(jl_dim)).astype(np.float32)

        self.buf_x       = np.zeros((max_memory, input_dim), dtype=np.float32)
        self.buf_y       = np.zeros(max_memory, dtype=np.int64)
        self.buf_proj    = np.zeros((max_memory, jl_dim),    dtype=np.float32)
        self.buf_weights = np.zeros(max_memory,              dtype=np.float32)
        self.buf_size    = 0
        self.write_ptr   = 0

        self.mu_t        = None
        self.sigma_t     = 1.0
        self.mu_baseline = None
        self.activated   = False

    def activate(self):
        self.activated = True

    def process_stream(self, x: np.ndarray, y: np.ndarray):
        if not self.activated or x.shape[0] == 0: return
        proj = x @ self.jl_matrix

        if self.buf_size == 0:
            self._write(x, y, proj)
            return

        valid_proj = self.buf_proj[:self.buf_size]
        dists      = np.sqrt(((proj[:, None] - valid_proj[None]) ** 2).sum(-1))
        d_min      = dists.min(1)
        nn_idx     = dists.argmin(1)

        cur_mu    = float(d_min.mean())
        cur_sigma = float(d_min.std()) + 1e-9
        if self.mu_baseline is None:
            self.mu_baseline = cur_mu
            self.mu_t, self.sigma_t = cur_mu, cur_sigma

        self.mu_t    = (1 - self.beta) * self.mu_t    + self.beta * cur_mu
        self.sigma_t = (1 - self.beta) * self.sigma_t + self.beta * cur_sigma

        drift   = abs(self.mu_t - self.mu_baseline) / (self.mu_baseline + 1e-9)
        gamma_t = self.base_gamma * (1 + self.kappa * drift)
        self.buf_weights[:self.buf_size] *= np.exp(-gamma_t)

        tau_t = self.mu_t + self.kappa * self.sigma_t
        novel = d_min > tau_t

        if novel.any(): self._write(x[novel], y[novel], proj[novel])
        for nn in nn_idx[~novel]:
            if nn < self.buf_size: self.buf_weights[nn] += 1.0

        if self.buf_size >= self.max_memory:
            worst = int(np.argmin(self.buf_weights[:self.buf_size]))
            pass # Circular overwrite handles eviction natively

    def _write(self, x, y, proj):
        for i in range(len(x)):
            idx = self.write_ptr % self.max_memory
            self.buf_x[idx], self.buf_y[idx], self.buf_proj[idx], self.buf_weights[idx] = x[i], y[i], proj[i], 1.0
            self.write_ptr += 1
            self.buf_size = min(self.buf_size + 1, self.max_memory)

    def get_coreset(self):
        return self.buf_x[:self.buf_size].copy(), self.buf_y[:self.buf_size].copy()

class SynergyClient:
    def __init__(self, cid: str, stream_train_x: np.ndarray, stream_train_y: np.ndarray,
                 model: SynergyModel, hparams: dict, stream_val_x: np.ndarray = None, stream_val_y: np.ndarray = None):
        self.cid = cid
        self.stream_x, self.stream_y = stream_train_x, stream_train_y
        self.val_x = stream_val_x if stream_val_x is not None else (stream_train_x if stream_train_x is not None else np.zeros((0, 0)))
        self.val_y = stream_val_y if stream_val_y is not None else (stream_train_y if stream_train_y is not None else np.zeros((0,)))
        self.model = model
        self.hparams = hparams
        d = stream_train_x.shape[1] if stream_train_x.shape[0] > 0 else 78
        jl_dim = min(16, d)
        self.sfc = SFCFilter(d, jl_dim, hparams.get('M_max', 500), hparams.get('gamma', 0.01))
        self.scaler = WelfordScaler(d)

    def get_lora_ndarrays(self): return self.model.get_lora_ndarrays()
    def set_lora_ndarrays(self, arrays): self.model.set_lora_ndarrays(arrays)

    def local_train(self, global_params, epochs: int = 1, lr: float = 1e-3, lambda_reg: float = 0.0,
                    batch_size: int = 32, r_k: int = 8, quantize_bits: int = 32):
        if global_params is not None: self.set_lora_ndarrays(global_params)

        bsz = 64
        for b in range(0, len(self.stream_x), bsz):
            bx, by = self.stream_x[b:b+bsz], self.stream_y[b:b+bsz]
            bi = b // bsz
            if bi < 5: self.scaler.partial_fit(bx)
            else:
                if not self.scaler.is_frozen:
                    self.scaler.freeze()
                    self.sfc.activate()
                self.sfc.process_stream(self.scaler.transform(bx), by)

        cx, cy = self.sfc.get_coreset()
        if len(cx) == 0: return self.get_lora_ndarrays(), 0, {"accuracy": 0.0, "f1": 0.0}, 0, 0

        self.model.train()
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        weight_decay = 1e-4

        ds = torch.utils.data.TensorDataset(torch.tensor(cx), torch.tensor(cy))
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

        for _ in range(epochs):
            for xb, yb in dl:
                for p in trainable:
                    if p.grad is not None:
                        p.grad.detach_(); p.grad.zero_()
                logits = self.model(xb, r_k)
                loss = torch.nn.functional.cross_entropy(logits, yb)
                ortho = sum(m.orthogonal_penalty(r_k) for m in self.model.modules() if hasattr(m, "orthogonal_penalty"))
                total_loss = loss + lambda_reg * ortho
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                for p in trainable:
                    if p.grad is None: continue
                    grad = p.grad.data
                    p.data.add_( -lr * grad - lr * weight_decay * p.data )
                    p.grad.detach_(); p.grad.zero_()

        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(cx), r_k)
            preds = logits.argmax(1).cpu().numpy()
        acc = float((preds == cy).mean())
        try:
            from sklearn.metrics import f1_score
            f1 = float(f1_score(cy, preds, average='macro'))
        except Exception: f1 = 0.0

        arrays = self.get_lora_ndarrays()
        total_elements = sum(a.size for a in arrays)
        bytes_before = total_elements * 4
        bytes_after = total_elements * (quantize_bits // 8) if quantize_bits < 32 else bytes_before

        metrics = {"accuracy": acc, "f1": f1, "n_coreset": float(len(cx))}
        return arrays, len(cx), metrics, bytes_before, bytes_after

    def local_eval(self, global_params=None, r_k: int = 8):
        if global_params is not None:
            try: self.set_lora_ndarrays(global_params)
            except Exception: pass

        eval_x, eval_y = getattr(self, 'val_x', None), getattr(self, 'val_y', None)
        if eval_x is None or len(eval_x) == 0: eval_x, eval_y = getattr(self, 'stream_x', None), getattr(self, 'stream_y', None)
        if eval_x is None or len(eval_x) == 0: return float(0.0), 1, {"accuracy": 0.0, "f1": 0.0}

        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(eval_x), r_k)
            loss = float(torch.nn.functional.cross_entropy(logits, torch.tensor(eval_y)).cpu().numpy())
            preds = logits.argmax(1).cpu().numpy()
        try:
            from sklearn.metrics import f1_score
            f1 = float(f1_score(eval_y, preds, average='macro'))
        except Exception: f1 = 0.0
        acc = float((preds == eval_y).mean())
        return loss, max(1, len(eval_x)), {"accuracy": acc, "f1": f1}

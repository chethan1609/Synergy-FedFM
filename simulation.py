import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from dataset_loader import read_and_clean_csvs
import csv
import time
import pathlib
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None
from typing import Dict, Tuple
from model import SynergyModel
from client import SynergyClient

def compute_central_baseline(X: np.ndarray, y: np.ndarray, num_classes: int, num_features: int, hparams: dict,
                             epochs: int = 15, lr: float = 0.05, batch_size: int = 64) -> float:
    from sklearn.metrics import f1_score
    n = len(X)
    if n == 0: return 0.0
    perm = np.random.permutation(n)
    test_size = max(1, int(0.1 * n))
    test_idx, train_idx = perm[:test_size], perm[test_size:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    print(f"[BASELINE DEBUG] n={n}, X_train.shape={X_train.shape}, X_test.shape={X_test.shape}")

    model = SynergyModel(num_features, hparams.get('d_hidden', 128), num_classes, hparams.get('R_max', 16))
    try:
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
        model.train()
        ds = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
        for _ in range(epochs):
            for xb, yb in dl:
                opt.zero_grad()
                logits = model(xb, hparams.get('R_max', 16))
                # Class-weighted loss to address imbalance
                try:
                    class_counts = np.bincount(y_train.astype(int))
                    weights = torch.tensor(1.0 / (class_counts + 1e-9), dtype=torch.float32)
                    weights = weights / weights.sum()
                    weights = weights.to(logits.device)
                    loss = torch.nn.functional.cross_entropy(logits, yb, weight=weights)
                except Exception:
                    loss = torch.nn.functional.cross_entropy(logits, yb)
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X_test), hparams.get('R_max', 16))
            preds = logits.argmax(1).cpu().numpy()
        try:
            score = float(f1_score(y_test, preds, average='macro'))
            print(f"[BASELINE DEBUG] computed baseline f1={score:.6f}")
            return score
        except Exception as e:
            print(f"[BASELINE DEBUG] f1_score exception: {e}")
            return 0.0
    except Exception as e:
        print(f"[BASELINE DEBUG] training exception: {e}")
        return 0.0

def partition_data(X: np.ndarray, y: np.ndarray, num_clients: int, alpha: float,
                   val_frac: float = 0.1, test_frac: float = 0.1) -> Dict[str, dict]:
    global_idx = np.arange(len(X))
    client_data = {str(i): {"x": [], "y": [], "idx": []} for i in range(num_clients)}
    for c in np.unique(y):
        c_idx = np.where(y == c)[0]
        props = np.random.dirichlet(np.repeat(alpha, num_clients))
        splits = (np.cumsum(props) * len(c_idx)).astype(int)[:-1]
        for cid, idx_slice in enumerate(np.split(c_idx, splits)):
            if len(idx_slice):
                client_data[str(cid)]["x"].append(X[idx_slice])
                client_data[str(cid)]["y"].append(y[idx_slice])
                client_data[str(cid)]["idx"].append(global_idx[idx_slice])
    streams = {}
    for cid, d in client_data.items():
        if not d["x"]:
            streams[cid] = {"train": (np.zeros((0, X.shape[1]), np.float32), np.zeros((0,), np.int64)),
                            "val": (np.zeros((0, X.shape[1]), np.float32), np.zeros((0,), np.int64)),
                            "test": (np.zeros((0, X.shape[1]), np.float32), np.zeros((0,), np.int64))}
            continue
        cx = np.concatenate(d["x"]) ; cy = np.concatenate(d["y"]) ; ci = np.concatenate(d["idx"]) ; order = np.argsort(ci)
        cx_all, cy_all = cx[order].astype(np.float32), cy[order].astype(np.int64)
        n = len(cx_all)
        perm = np.random.permutation(n)
        cx_all, cy_all = cx_all[perm], cy_all[perm]
        n_val = int(max(1, np.floor(val_frac * n))) if n > 1 else 0
        n_test = int(max(1, np.floor(test_frac * n))) if n > 1 else 0
        n_train = n - n_val - n_test
        if n_train <= 0:
            n_train = max(1, n - n_test)
            if n_train + n_test > n: n_test = max(0, n - n_train)
        
        train_x = cx_all[:n_train] if n_train > 0 else np.zeros((0, X.shape[1]), np.float32)
        train_y = cy_all[:n_train] if n_train > 0 else np.zeros((0,), np.int64)
        val_x = cx_all[n_train:n_train + n_val] if n_val > 0 else np.zeros((0, X.shape[1]), np.float32)
        val_y = cy_all[n_train:n_train + n_val] if n_val > 0 else np.zeros((0,), np.int64)
        test_x = cx_all[n_train + n_val:] if n_test > 0 else np.zeros((0, X.shape[1]), np.float32)
        test_y = cy_all[n_train + n_val:] if n_test > 0 else np.zeros((0,), np.int64)
        streams[cid] = {"train": (train_x, train_y), "val": (val_x, val_y), "test": (test_x, test_y)}
    return streams

def run_flower_simulation(args, streams, num_classes, num_features, hparams):
    try:
        import flwr as fl
        from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
    except Exception as e:
        print("[SIMULATION] Failed importing Flower or dependencies:", e)
        return None

    class FlowerSynergyClient(fl.client.NumPyClient):
        def __init__(self, cid, train_x, train_y, model, hparams, val_x=None, val_y=None):
            self.inner = SynergyClient(cid, train_x, train_y, model, hparams, stream_val_x=val_x, stream_val_y=val_y)
            self.model = model
            self.hparams = hparams

        def get_parameters(self, config):
            return [v.cpu().numpy() for v in self.model.lora_state_dict().values()]

        def set_parameters(self, parameters):
            keys = [n for n, _ in self.model.named_parameters() if "lora" in n]
            with torch.no_grad():
                for k, arr in zip(keys, parameters):
                    self.model.state_dict()[k].copy_(torch.tensor(arr))

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            r_k = int(config.get("r_k", self.hparams.get("R_max", 16)))
            arrays, n_examples, metrics, _, _ = self.inner.local_train(
                global_params=None,
                epochs=self.hparams.get("epochs", 5),
                lr=self.hparams.get("lr", 0.05),
                lambda_reg=self.hparams.get("lambda_reg", 0.1),
                batch_size=self.hparams.get("batch_size", 64),
                r_k=r_k,
                quantize_bits=self.hparams.get("quantize_bits", 8),
            )
            if n_examples == 0:
                return self.get_parameters({}), 1, {"accuracy": 0.0, "f1": 0.0}
            return arrays, n_examples, metrics

        def evaluate(self, parameters, config):
            try:
                loss, n_examples, metrics = self.inner.local_eval(parameters, r_k=self.hparams.get('r_k', self.hparams.get('R_max', 16)))
            except Exception as e:
                print(f"[Flower Client] evaluate() failed: {e}")
                return 0.0, 1, {"accuracy": 0.0}
            return float(loss), max(1, int(n_examples)), metrics

    template = SynergyModel(num_features, hparams.get('d_hidden', 128), num_classes, hparams.get('R_max', 16))
    total_params = sum(p.numel() for p in template.parameters())
    trainable_params = sum(p.numel() for p in template.parameters() if p.requires_grad)
    C_compute_const = float(total_params) / float(trainable_params if trainable_params > 0 else 1)
    num_clients = len(streams)

    def client_fn(cid: str):
        model = SynergyModel(num_features, hparams.get('d_hidden', 128), num_classes, hparams.get('R_max', 16))
        s = streams.get(cid, None)
        if s is None:
            train_x = np.zeros((0, num_features), np.float32)
            train_y, val_y = np.zeros((0,), np.int64), np.zeros((0,), np.int64)
            val_x = np.zeros((0, num_features), np.float32)
        else:
            train_x, train_y = s.get('train', (np.zeros((0, num_features), np.float32), np.zeros((0,), np.int64)))
            val_x, val_y = s.get('val', (np.zeros((0, num_features), np.float32), np.zeros((0,), np.int64)))
        return FlowerSynergyClient(cid, train_x, train_y, model, dict(hparams), val_x=val_x, val_y=val_y)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0, fraction_evaluate=1.0,
        min_fit_clients=min(5, num_clients), min_evaluate_clients=min(3, num_clients), min_available_clients=num_clients,
        initial_parameters=ndarrays_to_parameters(template.get_lora_ndarrays()),
    )

    exp_name = args.get("exp_name", f"exp_{int(time.time())}")
    results_dir = pathlib.Path(args.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_file = results_dir / f"{exp_name}.csv"
    if not csv_file.exists():
        with open(csv_file, "w", newline="") as f:
            csv.writer(f).writerow(["round", "accuracy", "f1", "comm_bytes", "compression", "CCR", "delta_f1", "timestamp"])

    raw_total_examples = 0
    for s in streams.values():
        if s:
            for part in ('train', 'val', 'test'): raw_total_examples += int(len(s.get(part, ([], []))[0]))

    try:
        # Allow callers to force the local emulation path for stability/debugging
        if isinstance(args, dict) and args.get('force_emulation', False):
            raise Exception("forced emulation")
        history = fl.simulation.start_simulation(
            client_fn=client_fn, num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=args.get('num_rounds', 3)),
            strategy=strategy, client_resources={"num_cpus": 1, "num_gpus": 0.0},
        )
        return history
    except Exception as e:
        print(f"[SIMULATION] Flower simulation unavailable; running local FedAvg emulation: {e}")
        
        num_rounds = args.get('num_rounds', 3)
        global_params = template.get_lora_ndarrays()

        def aggregate_fit(all_arrays, all_n):
            total = max(1, sum(all_n))
            return [sum(a[layer_idx] * n for a, n in zip(all_arrays, all_n)) / total for layer_idx in range(len(all_arrays[0]))]

        history = {"metrics_distributed": {"accuracy": []}}
        for rnd in range(1, num_rounds + 1):
            print(f"[ROUND {rnd}] Running local FedAvg emulation")
            all_arrays, all_n, accs, f1s = [], [], [], []
            bytes_sent, bytes_sent_after = 0, 0

            for cid in range(num_clients):
                model = SynergyModel(num_features, hparams.get('d_hidden', 128), num_classes, hparams.get('R_max', 16))
                s = streams.get(str(cid), None)
                if s is None:
                    train_x = np.zeros((0, num_features), np.float32)
                    train_y, val_y = np.zeros((0,), np.int64), np.zeros((0,), np.int64)
                    val_x = np.zeros((0, num_features), np.float32)
                else:
                    train_x, train_y = s.get('train', (np.zeros((0, num_features), np.float32), np.zeros((0,), np.int64)))
                    val_x, val_y = s.get('val', (np.zeros((0, num_features), np.float32), np.zeros((0,), np.int64)))
                client = SynergyClient(str(cid), train_x, train_y, model, hparams, stream_val_x=val_x, stream_val_y=val_y)
                params, n_examples, metrics, b_before, b_after = client.local_train(
                    global_params,
                    epochs=hparams.get('epochs', 5), lr=hparams.get('lr', 0.05),
                    lambda_reg=hparams.get('lambda_reg', 0.1), batch_size=hparams.get('batch_size', 64),
                    r_k=hparams.get('r_k', hparams.get('R_max', 16)), quantize_bits=hparams.get('quantize_bits', 8),
                )
                all_arrays.append(params); all_n.append(n_examples if n_examples is not None else 0)
                accs.append(metrics.get('accuracy', 0.0)); f1s.append(metrics.get('f1', 0.0))
                bytes_sent += b_before; bytes_sent_after += b_after

            if sum(all_n) == 0: break

            global_params = aggregate_fit(all_arrays, all_n)
            mean_acc, mean_f1 = float(np.mean(accs)) if accs else 0.0, float(np.mean(f1s)) if f1s else 0.0
            avg_comm = bytes_sent_after / max(1, num_clients)
            compression = (bytes_sent / bytes_sent_after) if bytes_sent_after > 0 else 1.0
            coreset_total = max(1, sum(all_n))
            C_data = float(raw_total_examples) / float(coreset_total)
            CCR = float(C_data) * float(C_compute_const) * float(compression)
            # DEBUG: print baseline vs federated mean to ensure baseline_f1 is passed correctly
            baseline_val = float(args.get('baseline_f1', 0.0))
            delta_f1 = float(baseline_val - mean_f1)
            print(f"[DEBUG] baseline_f1={baseline_val:.6f}  mean_f1={mean_f1:.6f}  delta_f1={delta_f1:.6f}")
            history["metrics_distributed"]["accuracy"].append((rnd, mean_acc))
            print(f"[ROUND {rnd}] mean_accuracy={mean_acc:.4f} mean_f1={mean_f1:.4f} avg_comm_bytes={avg_comm:.0f} compression={compression:.2f}")

            with open(csv_file, "a", newline="") as f:
                csv.writer(f).writerow([rnd, mean_acc, mean_f1, int(avg_comm), float(compression), float(CCR), float(delta_f1), int(time.time())])

        return history

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--parquet', default='./cicids2017_clean.parquet')
    parser.add_argument('--num_clients', type=int, default=4)
    parser.add_argument('--num_rounds', type=int, default=3)
    args = parser.parse_args()
    try:
        df = pd.read_parquet(args.parquet)
    except Exception as e:
        print(f"[SIMULATION] Failed to read parquet {args.parquet}. Checking for raw dataset folder...")
        df = read_and_clean_csvs("./dataset")
    labels = df['Label'].values
    classes, encoded = np.unique(labels, return_inverse=True)
    X = df.drop(columns=['Label']).values.astype(np.float32)
    streams = partition_data(X, encoded, args.num_clients, 0.1)
    
    hparams = {
        'd_hidden': 128, 'R_max': 16, 'r_k': 16, 
        'epochs': 5, 'lr': 0.05, 'lambda_reg': 0.1, 
        'batch_size': 64, 'quantize_bits': 8, 
        'M_max': 500, 'gamma': 0.01
    }
    
    try: baseline_f1 = compute_central_baseline(X, encoded, len(classes), X.shape[1], hparams, epochs=50, lr=0.01, batch_size=64)
    except Exception as e:
        print(f"[SIMULATION] central baseline failed: {e}")
        baseline_f1 = 0.0
        
    args_dict = dict(vars(args))
    args_dict['baseline_f1'] = float(baseline_f1)
    print(f"[SIMULATION] baseline_f1={baseline_f1:.4f}")
    history = run_flower_simulation(args_dict, streams, len(classes), X.shape[1], hparams)
    print('Done')

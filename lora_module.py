import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ElasticLoRALayer(nn.Module):
    """LoRA-style linear layer with configurable max rank r_max."""
    def __init__(self, in_features: int, out_features: int, r_max: int):
        super().__init__()
        self.base_weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.01, requires_grad=False)
        
        # Fixed Initialization: Ensures gradients flow immediately
        self.lora_A = nn.Parameter(torch.randn(in_features, r_max) * 0.1)
        self.lora_B = nn.Parameter(torch.randn(r_max, out_features) * 0.1)
        self.r_max = r_max

    def forward(self, x: torch.Tensor, r: int) -> torch.Tensor:
        r = min(r, self.r_max)
        return F.linear(x, self.base_weight) + (x @ self.lora_A[:, :r]) @ self.lora_B[:r, :]

    def orthogonal_penalty(self, r: int) -> torch.Tensor:
        r = min(r, self.r_max)
        A = self.lora_A[:, :r]
        gram = A.T @ A
        return torch.norm(gram - torch.eye(r, device=gram.device), p='fro') ** 2

    def get_lora_state(self):
        return {"lora_A": self.lora_A.detach().cpu().numpy(),
                "lora_B": self.lora_B.detach().cpu().numpy()}

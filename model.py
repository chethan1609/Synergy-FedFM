import torch
import torch.nn as nn
from typing import Dict, List
from lora_module import ElasticLoRALayer

class SynergyModel(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, r_max: int):
        super().__init__()
        self.l1 = ElasticLoRALayer(d_in, d_hidden, r_max)
        self.l2 = ElasticLoRALayer(d_hidden, d_out, r_max)

    def forward(self, x: torch.Tensor, r: int):
        return self.l2(torch.relu(self.l1(x, r)), r)

    def lora_state_dict(self) -> Dict[str, torch.Tensor]:
        return {n: p.data.clone() for n, p in self.named_parameters() if "lora" in n}

    def get_lora_ndarrays(self) -> List:
        return [v.cpu().numpy() for v in self.lora_state_dict().values()]

    def set_lora_ndarrays(self, arrays: List):
        keys = [n for n, _ in self.named_parameters() if "lora" in n]
        with torch.no_grad():
            for k, arr in zip(keys, arrays):
                self.state_dict()[k].copy_(torch.tensor(arr))

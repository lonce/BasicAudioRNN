import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
import random


@dataclass
class GRUAudioConfig:
    num_conditioning_params: int  # number of real-valued params per input
    hidden_size: int = 128        # GRU hidden size
    num_layers: int = 2           # number of GRU layers
    dropout: float = 0.1          # dropout between GRU layers


class GRUAudioModel(nn.Module):
    def __init__(self, config: GRUAudioConfig):
        super().__init__()
        self.config = config

        input_size = 1 + config.num_conditioning_params  # 1 for mu-law input as float
        self.input_proj = nn.Linear(input_size, config.hidden_size)

        self.gru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )

        self.output_proj = nn.Linear(config.hidden_size, 256)

        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

    def forward(self, mu_law_input: torch.FloatTensor, cond_params: torch.FloatTensor):
        """
        mu_law_input: FloatTensor of shape [B, T] with values in [0, 1]
        cond_params: FloatTensor of shape [B, T, p] with values in [0,1]
        Returns: logits of shape [B, T, 256]
        """
        x = mu_law_input.unsqueeze(-1)  # [B, T, 1]
        x = torch.cat([x, cond_params], dim=-1)  # [B, T, p+1]
        x = self.input_proj(x)                   # [B, T, hidden_size]
        gru_out, _ = self.gru(x)                 # [B, T, hidden_size]
        logits = self.output_proj(gru_out)       # [B, T, 256]
        return logits


# def top_n_sample(logits: torch.Tensor, n: int = 3) -> torch.Tensor:
#     """
#     Sample from top-n of the logits distribution.
#     logits: Tensor of shape [B, 256] (unnormalized)
#     Returns: Tensor of shape [B] with sampled class indices
#     """
#     probs = F.softmax(logits, dim=-1)                       # [B, 256]
#     top_probs, top_indices = torch.topk(probs, n, dim=-1)  # [B, n], [B, n]

#     # Normalize top-n probs
#     top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)

#     # Sample from top-n
#     samples = []
#     for i in range(probs.size(0)):
#         sampled = random.choices(
#             population=top_indices[i].tolist(),
#             weights=top_probs[i].tolist(),
#             k=1
#         )[0]
#         samples.append(sampled)

#     return torch.tensor(samples, device=logits.device, dtype=torch.long)

def top_n_sample(logits: torch.Tensor, n: int = 3, temperature: float = 1.0) -> torch.Tensor:
    """
    Sample from the top-n logits after applying temperature-scaled softmax.

    Args:
        logits (Tensor): [1, 256] unnormalized logits
        n (int): number of top logits to consider
        temperature (float): temperature for softmax smoothing

    Returns:
        Tensor of shape [1] containing the sampled index
    """
    probs = torch.softmax(logits / temperature, dim=-1)  # [1, 256]
    top_probs, top_indices = torch.topk(probs, n)        # [1, n]

    # Normalize top-n probabilities
    top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)  # [1, n]

    sampled = torch.multinomial(top_probs, num_samples=1)        # [1, 1]
    sampled_index = top_indices[0, sampled[0, 0]]                # scalar

    return torch.tensor([sampled_index.item()], device=logits.device, dtype=torch.long)

# def top_n_sample(logits: torch.Tensor, n: int = 3, temperature: float = 1.0) -> torch.Tensor:
#     """
#     Sample from the top-n logits after applying temperature-scaled softmax.

#     Args:
#         logits (Tensor): [1, 256] unnormalized logits
#         n (int): number of top logits to consider
#         temperature (float): temperature for softmax smoothing

#     Returns:
#         Tensor of shape [1] containing the selected class index
#     """
#     probs = torch.softmax(logits / temperature, dim=-1)
#     top_probs, top_indices = torch.topk(probs, n)
#     top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)  # re-normalize
#     sampled = torch.multinomial(top_probs, 1)
#     return top_indices.gather(-1, sampled)
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
import random


@dataclass
class GRUAudioConfig:
    num_conditioning_params: int  # number of real-valued params per input
    embedding_dim: int = 16       # dimension of mu-law embedding
    hidden_size: int = 128        # GRU hidden size
    num_layers: int = 2           # number of GRU layers
    dropout: float = 0.1          # dropout between GRU layers


class GRUAudioModel(nn.Module):
    def __init__(self, config: GRUAudioConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(256, config.embedding_dim)
        input_size = config.embedding_dim + config.num_conditioning_params
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

    def forward(self, mu_law_input: torch.LongTensor, cond_params: torch.FloatTensor):
        """
        mu_law_input: LongTensor of shape [B, T] with values in [0, 255]
        cond_params: FloatTensor of shape [B, T, p] with values in [0,1]
        Returns: logits of shape [B, T, 256]
        """
        emb = self.embedding(mu_law_input)                  # [B, T, embedding_dim]
        x = torch.cat([emb, cond_params], dim=-1)           # [B, T, embedding + p]
        x = self.input_proj(x)                              # [B, T, hidden_size]
        gru_out, _ = self.gru(x)                            # [B, T, hidden_size]
        logits = self.output_proj(gru_out)                  # [B, T, 256]
        return logits


def top_n_sample(logits: torch.Tensor, n: int = 3) -> torch.Tensor:
    """
    Sample from top-n of the logits distribution.
    logits: Tensor of shape [B, 256] (unnormalized)
    Returns: Tensor of shape [B] with sampled class indices
    """
    probs = F.softmax(logits, dim=-1)                       # [B, 256]
    top_probs, top_indices = torch.topk(probs, n, dim=-1)  # [B, n], [B, n]

    # Normalize top-n probs
    top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)

    # Sample from top-n
    samples = []
    for i in range(probs.size(0)):
        sampled = random.choices(
            population=top_indices[i].tolist(),
            weights=top_probs[i].tolist(),
            k=1
        )[0]
        samples.append(sampled)

    return torch.tensor(samples, device=logits.device, dtype=torch.long)

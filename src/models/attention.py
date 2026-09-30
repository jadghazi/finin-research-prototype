"""Paper section 4.2: headline interaction and market-query weighting."""

from __future__ import annotations

import math

import torch
from torch import nn


class NewsAttention(nn.Module):
    def __init__(self, width: int = 64):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(width, 1, batch_first=True)
        self.normalization = nn.LayerNorm(width)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)

    def forward(
        self,
        market: torch.Tensor,
        news: torch.Tensor,
        mask: torch.Tensor,
        use_self_attention: bool = True,
        use_market_attention: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not torch.all(mask.any(dim=1)):
            raise ValueError("Every example must have at least one real headline")
        if use_self_attention:
            attended, _ = self.self_attention(news, news, news, key_padding_mask=~mask, need_weights=False)
            refined = self.normalization(news + attended)
        else:
            refined = news
        if use_market_attention:
            scores = torch.sum(self.query(market).unsqueeze(1) * self.key(refined), dim=-1)
            scores = scores / math.sqrt(refined.shape[-1])
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=1)
        else:
            weights = mask.to(refined.dtype) / mask.sum(dim=1, keepdim=True)
        weights = weights.masked_fill(~mask, 0)
        news_summary = torch.sum(refined * weights.unsqueeze(-1), dim=1)
        return news_summary, weights

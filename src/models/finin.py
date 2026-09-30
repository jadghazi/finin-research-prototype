"""A one-input-day, small-width implementation of FININ Eqs. 1-7."""

from __future__ import annotations

import torch
from torch import nn

from src.models.attention import NewsAttention
from src.models.encoders import FusionEncoders


class ReducedFININ(nn.Module):
    def __init__(self, variant: str = "full", width: int = 64, dropout: float = 0.1):
        super().__init__()
        if variant not in {"full", "no_self_attention", "no_sentiment", "mean_pool"}:
            raise ValueError(f"Unknown FININ variant: {variant}")
        self.variant = variant
        self.encoders = FusionEncoders(width)
        self.attention = NewsAttention(width)
        self.predictor = nn.Sequential(
            nn.Linear(2 * width, width), nn.ReLU(), nn.Dropout(dropout), nn.Linear(width, 1)
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        market, news = self.encoders(
            batch["market_description"],
            batch["market_features"],
            batch["headline_vectors"],
            batch["sentiment"],
            use_sentiment=self.variant != "no_sentiment",
        )
        summary, weights = self.attention(
            market,
            news,
            batch["mask"],
            use_self_attention=self.variant not in {"no_self_attention", "mean_pool"},
            use_market_attention=self.variant != "mean_pool",
        )
        logits = self.predictor(torch.cat([market, summary], dim=-1)).squeeze(-1)
        return logits, weights

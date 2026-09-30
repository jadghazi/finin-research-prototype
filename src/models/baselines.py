"""Market-only and market-plus-mean-sentiment comparisons."""

from __future__ import annotations

import torch
from torch import nn


class NumericBaseline(nn.Module):
    def __init__(self, use_sentiment: bool, width: int = 64, dropout: float = 0.1):
        super().__init__()
        self.use_sentiment = use_sentiment
        self.predictor = nn.Sequential(
            nn.Linear(9 if use_sentiment else 6, width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, None]:
        features = batch["market_features"]
        if self.use_sentiment:
            mask = batch["mask"].unsqueeze(-1)
            average = (batch["sentiment"] * mask).sum(dim=1) / mask.sum(dim=1)
            features = torch.cat([features, average], dim=-1)
        return self.predictor(features).squeeze(-1), None

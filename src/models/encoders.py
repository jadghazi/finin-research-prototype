"""Paper Eqs. 1-5: shared text projection, numerical encoders and fusion."""

from __future__ import annotations

import torch
from torch import nn


def mlp(input_width: int, output_width: int = 64) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_width, output_width), nn.ReLU(), nn.Linear(output_width, output_width))


class FusionEncoders(nn.Module):
    def __init__(self, width: int = 64):
        super().__init__()
        self.text = nn.Linear(384, width)
        self.market_numbers = mlp(6, width)
        self.news_numbers = mlp(3, width)
        self.market_fusion = mlp(2 * width, width)
        self.news_fusion = mlp(2 * width, width)

    def forward(
        self,
        market_description: torch.Tensor,
        market_features: torch.Tensor,
        headline_vectors: torch.Tensor,
        sentiment: torch.Tensor,
        use_sentiment: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        market_text = self.text(market_description).expand(market_features.shape[0], -1)
        market = self.market_fusion(torch.cat([market_text, self.market_numbers(market_features)], dim=-1))
        news_text = self.text(headline_vectors)
        news_numbers = self.news_numbers(sentiment if use_sentiment else torch.zeros_like(sentiment))
        news = self.news_fusion(torch.cat([news_text, news_numbers], dim=-1))
        return market, news

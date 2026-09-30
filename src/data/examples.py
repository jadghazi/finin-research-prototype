"""Create dated examples without moving future prices or source answers into features."""

from __future__ import annotations

import hashlib
import math
import statistics


MAX_HEADLINES = 16
MARKET_DESCRIPTION = "SPY is an exchange-traded fund tracking the S&P 500, a broad index of large U.S. companies."
MARKET_FEATURES = ("open_return", "high_return", "low_return", "close_return", "adjusted_return", "log_volume")
BOUNDARIES = {"train": "2017-06-28", "valid": "2019-02-13", "test": "9999-12-31"}


def select_headlines(news: str, limit: int = MAX_HEADLINES) -> list[dict[str, str]]:
    unique = {headline.strip() for headline in news.splitlines() if headline.strip()}
    if not unique:
        raise ValueError("A dated example has no headlines")
    ordered = sorted(unique, key=lambda headline: (hashlib.sha256(headline.encode("utf-8")).hexdigest(), headline))
    return [
        {"id": hashlib.sha256(headline.encode("utf-8")).hexdigest(), "text": headline}
        for headline in ordered[:limit]
    ]


def market_features(previous: dict[str, float], current: dict[str, float]) -> list[float]:
    previous_close = previous["close"]
    return [
        current["open"] / previous_close - 1,
        current["high"] / previous_close - 1,
        current["low"] / previous_close - 1,
        current["close"] / previous_close - 1,
        current["adj_close"] / previous["adj_close"] - 1,
        math.log1p(current["volume"]),
    ]


def build_examples(
    news_by_split: dict[str, list[dict]],
    sessions: list[str],
    prices: dict[str, dict[str, float]],
) -> tuple[dict[str, list[dict]], list[dict]]:
    positions = {day: index for index, day in enumerate(sessions)}
    prepared: dict[str, list[dict]] = {}
    exclusions: list[dict] = []
    for split, rows in news_by_split.items():
        examples = []
        for row in rows:
            news_date = row["date"]
            if news_date not in positions:
                raise ValueError(f"News date absent from SPY trading calendar: {news_date}")
            position = positions[news_date]
            if position + 2 >= len(sessions):
                raise ValueError(f"No next-session forecast and target for {news_date}")
            forecast_date = sessions[position + 1]
            target_date = sessions[position + 2]
            if target_date >= BOUNDARIES[split]:
                exclusions.append({"split": split, "news_date": news_date, "reason": "target crosses next split boundary"})
                continue
            current = prices[forecast_date]
            future = prices[target_date]
            headlines = select_headlines(row["news"])
            examples.append({
                "id": f"{split}_{news_date}",
                "split": split,
                "news_date": news_date,
                "forecast_date": forecast_date,
                "target_date": target_date,
                "market_description": MARKET_DESCRIPTION,
                "market_features_raw": market_features(prices[news_date], current),
                "headlines": headlines,
                "close_at_forecast": current["close"],
                "close_at_target": future["close"],
                "target_up": int(future["close"] > current["close"]),
            })
        prepared[split] = examples
    return prepared, exclusions


def fit_standardizer(training_examples: list[dict]) -> tuple[list[float], list[float]]:
    columns = list(zip(*(row["market_features_raw"] for row in training_examples)))
    mean = [statistics.fmean(column) for column in columns]
    scale = [statistics.pstdev(column) for column in columns]
    if any(value <= 0 or not math.isfinite(value) for value in scale):
        raise ValueError("Market feature has invalid training scale")
    return mean, scale


def apply_standardizer(examples: dict[str, list[dict]], mean: list[float], scale: list[float]) -> None:
    for rows in examples.values():
        for row in rows:
            row["market_features"] = [
                (value - center) / width
                for value, center, width in zip(row["market_features_raw"], mean, scale)
            ]

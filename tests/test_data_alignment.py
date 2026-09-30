"""Check the time boundary and labeling risks that can invalidate an experiment."""

from __future__ import annotations

import unittest

from src.data.examples import build_examples, fit_standardizer, select_headlines


def price(close: float, volume: float = 100.0) -> dict[str, float]:
    return {
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "adj_close": close,
        "volume": volume,
    }


class DataAlignmentTest(unittest.TestCase):
    def test_missing_news_session_does_not_turn_target_into_multiday_move(self) -> None:
        sessions = ["2015-01-05", "2015-01-06", "2015-01-07", "2015-01-08", "2015-01-09"]
        prices = dict(zip(sessions, [price(100), price(101), price(99), price(103), price(102)]))
        source = {
            "train": [{"date": "2015-01-05", "news": "headline A\nheadline B"},
                      {"date": "2015-01-07", "news": "headline C"}],
            "valid": [],
            "test": [],
        }
        examples, _ = build_examples(source, sessions, prices)
        first = examples["train"][0]
        self.assertEqual((first["news_date"], first["forecast_date"], first["target_date"]),
                         ("2015-01-05", "2015-01-06", "2015-01-07"))
        self.assertEqual(first["target_up"], 0)
        self.assertEqual(examples["train"][1]["target_date"], "2015-01-09")

    def test_boundary_excludes_future_outcome(self) -> None:
        sessions = ["2017-06-26", "2017-06-27", "2017-06-28", "2017-06-29"]
        prices = {day: price(100 + i) for i, day in enumerate(sessions)}
        rows = {"train": [{"date": "2017-06-26", "news": "A"}], "valid": [], "test": []}
        examples, excluded = build_examples(rows, sessions, prices)
        self.assertEqual(examples["train"], [])
        self.assertEqual(excluded[0]["news_date"], "2017-06-26")

    def test_headline_selection_ignores_source_order_and_duplicates(self) -> None:
        a = select_headlines(" A \nB\nA\nC", 2)
        b = select_headlines("C\nB\nA", 2)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 2)

    def test_standardizer_uses_only_training_rows(self) -> None:
        training = [{"market_features_raw": [1.0, 2.0]}, {"market_features_raw": [3.0, 6.0]}]
        mean, scale = fit_standardizer(training)
        self.assertEqual(mean, [2.0, 4.0])
        self.assertEqual(scale, [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()

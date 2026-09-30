"""Stage 1: freeze SPY prices and create audited, dated FININ examples."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.examples import MARKET_FEATURES, apply_standardizer, build_examples, fit_standardizer  # noqa: E402
from src.data.nifty import load_nifty, sha256_file  # noqa: E402
from src.data.prices import fetch_prices, load_prices  # noqa: E402


EXPECTED = {"train": (1475, 23592), "valid": (315, 5040), "test": (317, 5041)}


def verify_source_returns(news_by_split: dict[str, list[dict]], sessions: list[str], prices: dict) -> None:
    position = {day: index for index, day in enumerate(sessions)}
    checked = 0
    for rows in news_by_split.values():
        for row in rows:
            day = row["date"]
            next_day = sessions[position[day] + 1]
            observed = prices[next_day]["close"] / prices[day]["close"] - 1
            if abs(observed - float(row["pct_change"])) > 0.000051:
                raise ValueError(f"NIFTY return disagrees with independent SPY prices on {day}")
            checked += 1
    print(f"Verified supplied return timing: {checked}/{checked} rows match next trading session")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    news = load_nifty(ROOT)
    fetch_prices(ROOT)
    sessions, prices = load_prices(ROOT)
    verify_source_returns(news, sessions, prices)
    examples, exclusions = build_examples(news, sessions, prices)
    means, scales = fit_standardizer(examples["train"])
    apply_standardizer(examples, means, scales)

    processed = ROOT / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split, rows in examples.items():
        slots = sum(len(row["headlines"]) for row in rows)
        if (len(rows), slots) != EXPECTED[split]:
            raise ValueError(f"Unexpected {split} counts {(len(rows), slots)}, expected {EXPECTED[split]}")
        if any(not row["news_date"] < row["forecast_date"] < row["target_date"] for row in rows):
            raise ValueError(f"Time order failed in {split}")
        if any(row["target_up"] != int(row["close_at_target"] > row["close_at_forecast"]) for row in rows):
            raise ValueError(f"Target validation failed in {split}")
        output = processed / f"{split}.jsonl"
        write_jsonl(output, rows)
        summary[split] = {
            "examples": len(rows),
            "headline_slots": slots,
            "up_targets": sum(row["target_up"] for row in rows),
            "first_news_date": rows[0]["news_date"],
            "last_news_date": rows[-1]["news_date"],
            "sha256": sha256_file(output),
        }
        print(f"{split}: {len(rows)} examples, {slots} headline slots, {summary[split]['up_targets']} up targets")

    manifest = {
        "protocol": "Previous trading day's NIFTY news + market through forecast close -> next trading session's close direction",
        "target_formula": "int(close[target_date] > close[forecast_date])",
        "market_feature_names": MARKET_FEATURES,
        "market_train_mean": means,
        "market_train_scale": scales,
        "news_manifest_sha256": sha256_file(ROOT / "data" / "nifty_manifest.json"),
        "price_manifest_sha256": sha256_file(ROOT / "data" / "spy_manifest.json"),
        "split_summary": summary,
        "excluded": exclusions,
    }
    (processed / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    report = ROOT / "reports" / "data_preparation.md"
    lines = [
        "# Stage 1: prepared examples",
        "",
        "The input is a dated set of NIFTY headlines plus SPY prices through the forecast day's close. The answer is whether the following trading session closes higher.",
        "",
        "| Split | Examples | Selected headline slots | Up targets |",
        "| --- | ---: | ---: | ---: |",
    ]
    for split in ("train", "valid", "test"):
        s = summary[split]
        lines.append(f"| {split} | {s['examples']:,} | {s['headline_slots']:,} | {s['up_targets']:,} |")
    lines += [
        "",
        "## What one example means",
        "",
        "The headline text is selected by a fixed SHA-256 rule after whitespace trimming and exact within-day deduplication. The model never sees the future closing price, the NIFTY answer field, or the NIFTY prompt.",
        "",
        "| News date | Forecast date: last available prices | Target date: answer observed | Close at forecast | Close at target | Target up? | Headlines selected |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for split in ("train", "valid", "test"):
        row = examples[split][0]
        lines.append(
            f"| {row['news_date']} | {row['forecast_date']} | {row['target_date']} | "
            f"{row['close_at_forecast']:.2f} | {row['close_at_target']:.2f} | "
            f"{row['target_up']} | {len(row['headlines'])} |"
        )
    lines += [
        "",
        "## Checks",
        "",
        "- All 2,111 supplied NIFTY returns were checked against the frozen SPY price calendar. They align with the next trading session. Binary targets were calculated independently for this delayed-news experiment.",
        "- Forecast and target dates use the real trading calendar. Date ordering and every binary target were asserted for all retained examples.",
        "- Four examples were excluded because their target dates cross into the next source split. The exclusions and file checksums are in `data/processed/manifest.json`.",
        "- Six market features are standardized using training examples only. The mean and scale are in `data/processed/manifest.json`.",
        "- NIFTY provides dates without individual headline release times. We use headlines dated the previous trading day, but cannot prove precise historical publication availability.",
        "",
        "Next stage: benchmark frozen BGE text and financial TinyBERT sentiment on 100 selected headlines before caching all selected features.",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report: {report.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

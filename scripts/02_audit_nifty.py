"""Audit the raw NIFTY JSONL files before choosing a prediction target."""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from collections import Counter
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "nifty"
MANIFEST_PATH = ROOT / "data" / "nifty_manifest.json"
REPORT_PATH = ROOT / "reports" / "data_audit.md"
REQUIRED_FIELDS = {"date", "news", "label", "pct_change", "context", "conversations"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_split(path: Path) -> dict:
    dates: list[date] = []
    news_counts: list[int] = []
    labels: Counter[str] = Counter()
    duplicate_headlines = 0
    empty_news_days = 0
    header_only_contexts = 0
    prompt_price_row_counts: list[int] = []
    missing_pct_change = 0
    weekend_dates = 0

    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path.name}:{line_number}: invalid JSON") from error
            missing = REQUIRED_FIELDS - row.keys()
            if missing:
                raise ValueError(f"{path.name}:{line_number}: missing {sorted(missing)}")
            day = date.fromisoformat(row["date"])
            dates.append(day)
            weekend_dates += day.weekday() >= 5
            headlines = [part.strip() for part in row["news"].splitlines() if part.strip()]
            news_counts.append(len(headlines))
            empty_news_days += not headlines
            duplicate_headlines += len(headlines) - len(set(headlines))
            labels[str(row["label"])] += 1
            header_only_contexts += "\n" not in row["context"].strip()
            prompt = row["conversations"][0]["value"] if row["conversations"] else ""
            # Some prompts place headlines in the Context section too. Count
            # dated numeric CSV records, not arbitrary section lines.
            price_row_count = 0
            for part in prompt.splitlines():
                if not re.match(r"^\d{4}-\d{2}-\d{2},", part):
                    continue
                fields = part.split(",")
                if len(fields) != 16:
                    raise ValueError(f"{path.name}:{line_number}: malformed price row")
                date.fromisoformat(fields[0])
                tuple(float(value) for value in fields[1:])
                price_row_count += 1
            prompt_price_row_counts.append(price_row_count)
            missing_pct_change += row["pct_change"] is None

    if not dates:
        raise ValueError(f"{path.name}: no records")
    if dates != sorted(dates):
        raise ValueError(f"{path.name}: dates are not chronological")
    if len(dates) != len(set(dates)):
        raise ValueError(f"{path.name}: duplicate dates")

    return {
        "rows": len(dates),
        "first": dates[0].isoformat(),
        "last": dates[-1].isoformat(),
        "dates": set(dates),
        "min_news": min(news_counts),
        "median_news": statistics.median(news_counts),
        "max_news": max(news_counts),
        "total_news": sum(news_counts),
        "duplicate_headlines_within_day": duplicate_headlines,
        "empty_news_days": empty_news_days,
        "header_only_contexts": header_only_contexts,
        "prompt_price_rows_min": min(prompt_price_row_counts),
        "prompt_price_rows_median": statistics.median(prompt_price_row_counts),
        "prompt_price_rows_max": max(prompt_price_row_counts),
        "prompts_without_price_rows": sum(count == 0 for count in prompt_price_row_counts),
        "missing_pct_change": missing_pct_change,
        "weekend_dates": weekend_dates,
        "labels": dict(sorted(labels.items())),
    }


def main() -> None:
    if not MANIFEST_PATH.exists():
        raise SystemExit("Run python scripts/01_download_nifty.py first.")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    summary = {}
    for split in ("train", "valid", "test"):
        filename = f"{split}.jsonl"
        path = RAW_DIR / filename
        if not path.exists():
            raise SystemExit(f"Missing {path}; rerun the downloader.")
        if sha256_file(path) != manifest["files"][filename]["sha256"]:
            raise SystemExit(f"Checksum mismatch: {path}")
        summary[split] = audit_split(path)

    for earlier, later in (("train", "valid"), ("valid", "test")):
        if max(summary[earlier]["dates"]) >= min(summary[later]["dates"]):
            raise ValueError(f"{earlier}/{later} splits are not strictly chronological")
        if summary[earlier]["dates"] & summary[later]["dates"]:
            raise ValueError(f"{earlier}/{later} splits have overlapping dates")

    lines = [
        "# NIFTY raw data audit",
        "",
        f"Source: [{manifest['dataset_card']}]({manifest['dataset_card']})",
        f"Pinned revision: `{manifest['revision']}`",
        f"Downloaded (UTC): `{manifest['downloaded_at_utc']}`",
        "",
        "This audit describes the downloaded source files only. It does **not** establish that the published labels match FININ's binary target or that all headlines were available at a trading cutoff.",
        "",
        "| Split | Rows | First date | Last date | Headlines total | Min/day | Median/day | Max/day |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for split, result in summary.items():
        lines.append(
            f"| {split} | {result['rows']:,} | {result['first']} | {result['last']} | "
            f"{result['total_news']:,} | {result['min_news']} | {result['median_news']:g} | {result['max_news']} |"
        )

    lines.extend(["", "## Data quality checks", ""])
    for split, result in summary.items():
        lines.extend(
            [
                f"### {split}",
                "",
                f"- Labels: `{json.dumps(result['labels'], sort_keys=True)}`",
                f"- Empty-news dates: {result['empty_news_days']:,}",
                f"- Duplicate headline strings within a date: {result['duplicate_headlines_within_day']:,}",
                f"- Context fields with only a header/one line: {result['header_only_contexts']:,}",
                f"- Market-history rows embedded in conversation prompt (min/median/max): {result['prompt_price_rows_min']}/{result['prompt_price_rows_median']:g}/{result['prompt_price_rows_max']}",
                f"- Prompts with no embedded market-history rows: {result['prompts_without_price_rows']:,}",
                f"- Missing `pct_change`: {result['missing_pct_change']:,}",
                f"- Weekend dates: {result['weekend_dates']:,}",
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation for the next step",
            "",
            "- Dates are ordered and non-overlapping across splits (the audit would fail otherwise).",
            "- The revised plan specifies exact within-day deduplication and a deterministic cap of 16 headlines per example.",
            "- A separate full-data price/target check is documented in reports/plan_verification.md; this raw audit does not rerun that independent check.",
            "- That check found all 2,111 supplied returns match next-trading-day SPY returns. Earlier documentation claiming current-day alignment was incorrect. The three-class label still differs from FININ's binary target.",
            "- The revised plan derives binary targets from a separate verified SPY price snapshot and uses previous-trading-day news. Individual headline release-time availability remains uncertain.",
        ]
    )
    if all(result["header_only_contexts"] == result["rows"] for result in summary.values()):
        lines.append("- Every standalone `context` field contains only a header/one line. Market-history rows are instead embedded in the `conversations` prompt.")
    if summary["train"]["median_news"] >= 2 * summary["test"]["median_news"]:
        lines.append("- Headline volume falls sharply from training to testing. The experiment must report this coverage shift and should compare models on identical dates.")
    lines.append("")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report: {REPORT_PATH.relative_to(ROOT)}")
    for split, result in summary.items():
        print(f"{split}: {result['rows']:,} days, {result['total_news']:,} headlines, median {result['median_news']:g}/day")


if __name__ == "__main__":
    main()

"""Stage 2: benchmark then cache frozen language-model outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("benchmark", "cache"))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")

    from src.features.cache import benchmark, cache_all

    if args.action == "benchmark":
        result = benchmark(ROOT, args.batch_size)
        output = ROOT / "reports" / "feature_benchmark.json"
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Saved benchmark: {output.relative_to(ROOT)}")
    else:
        result = cache_all(ROOT, args.batch_size)
        print(f"Cached {result['headline_count']:,} unique headlines")


if __name__ == "__main__":
    main()

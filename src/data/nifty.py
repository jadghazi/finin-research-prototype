"""Read the pinned NIFTY source without using its answer fields as features."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


SPLITS = ("train", "valid", "test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_nifty(root: Path) -> dict[str, list[dict]]:
    manifest_path = root / "data" / "nifty_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result: dict[str, list[dict]] = {}
    for split in SPLITS:
        path = root / "data" / "raw" / "nifty" / f"{split}.jsonl"
        expected = manifest["files"][path.name]["sha256"]
        if sha256_file(path) != expected:
            raise ValueError(f"NIFTY checksum changed: {path}")
        with path.open(encoding="utf-8") as source:
            result[split] = [json.loads(line) for line in source]
        dates = [row["date"] for row in result[split]]
        if dates != sorted(set(dates)):
            raise ValueError(f"NIFTY {split} dates are not unique and ordered")
        if any(not row.get("news", "").strip() for row in result[split]):
            raise ValueError(f"NIFTY {split} has empty news")
    if result["train"][-1]["date"] >= result["valid"][0]["date"]:
        raise ValueError("Train/validation dates overlap")
    if result["valid"][-1]["date"] >= result["test"][0]["date"]:
        raise ValueError("Validation/test dates overlap")
    return result

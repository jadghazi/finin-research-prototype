"""Freeze and read a daily SPY price snapshot with an explicit trading calendar."""

from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


URL = "https://query1.finance.yahoo.com/v8/finance/chart/SPY?period1=1262304000&period2=1600992000&interval=1d"


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def fetch_prices(root: Path) -> None:
    destination = root / "data" / "raw" / "spy" / "daily.json"
    manifest_path = root / "data" / "spy_manifest.json"
    if destination.exists():
        content = destination.read_bytes()
        if not manifest_path.exists():
            raise ValueError("SPY snapshot exists without its manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if sha256_bytes(content) != manifest["sha256"]:
            raise ValueError("SPY snapshot differs from its manifest")
        print(f"Using verified local snapshot: {destination.relative_to(root)}")
        return

    request = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 FININ-research-prototype"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                content = response.read()
            parse_prices(content)
            break
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt == 2:
                raise RuntimeError("Could not download SPY prices after 3 attempts") from error
            time.sleep(2 * (attempt + 1))

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    dates, _ = parse_prices(content)
    manifest = {
        "source_url": URL,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sha256": sha256_bytes(content),
        "sessions": len(dates),
        "first_session": dates[0],
        "last_session": dates[-1],
        "raw_file": "data/raw/spy/daily.json",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved SPY snapshot: {len(dates)} sessions, {dates[0]} to {dates[-1]}")


def parse_prices(content: bytes) -> tuple[list[str], dict[str, dict[str, float]]]:
    response = json.loads(content)
    chart = response["chart"]
    if chart.get("error"):
        raise ValueError(f"Yahoo chart returned an error: {chart['error']}")
    result = chart["result"][0]
    quotes = result["indicators"]["quote"][0]
    adjusted = result["indicators"]["adjclose"][0]["adjclose"]
    dates: list[str] = []
    prices: dict[str, dict[str, float]] = {}
    for index, timestamp in enumerate(result["timestamp"]):
        day = datetime.fromtimestamp(timestamp, timezone.utc).date().isoformat()
        values = {key: quotes[key][index] for key in ("open", "high", "low", "close", "volume")}
        values["adj_close"] = adjusted[index]
        if any(value is None or not math.isfinite(value) for value in values.values()):
            raise ValueError(f"Missing or invalid price field for {day}")
        values = {key: float(value) for key, value in values.items()}
        if values["low"] > min(values["open"], values["close"]):
            raise ValueError(f"Invalid low price for {day}")
        if values["high"] < max(values["open"], values["close"]):
            raise ValueError(f"Invalid high price for {day}")
        if values["close"] <= 0 or values["adj_close"] <= 0 or values["volume"] <= 0:
            raise ValueError(f"Nonpositive price or volume for {day}")
        if day in prices:
            raise ValueError(f"Repeated Yahoo price date: {day}")
        dates.append(day)
        prices[day] = values
    if dates != sorted(dates):
        raise ValueError("Yahoo price dates not sorted")
    return dates, prices


def load_prices(root: Path) -> tuple[list[str], dict[str, dict[str, float]]]:
    destination = root / "data" / "raw" / "spy" / "daily.json"
    content = destination.read_bytes()
    manifest = json.loads((root / "data" / "spy_manifest.json").read_text(encoding="utf-8"))
    if sha256_bytes(content) != manifest["sha256"]:
        raise ValueError("SPY snapshot checksum changed")
    return parse_prices(content)

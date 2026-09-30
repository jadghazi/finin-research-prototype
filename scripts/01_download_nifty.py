"""Download one pinned NIFTY revision without a Hugging Face SDK dependency."""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = "raeidsaqur/NIFTY"
REVISION = "9b8aef736cbaf6a7e9645ac20e8bd9b2344734d2"
FILENAMES = ("train.jsonl", "valid.jsonl", "test.jsonl")
RAW_DIR = ROOT / "data" / "raw" / "nifty"
MANIFEST_PATH = ROOT / "data" / "nifty_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    previous_manifest = None
    if MANIFEST_PATH.exists():
        previous_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if previous_manifest.get("revision") != REVISION:
            raise SystemExit("Existing manifest refers to a different dataset revision.")
    files = {}

    for filename in FILENAMES:
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{filename}"
        destination = RAW_DIR / filename
        if destination.exists():
            print(f"Already present: {destination.relative_to(ROOT)}")
        else:
            temporary = destination.with_suffix(destination.suffix + ".part")
            request = urllib.request.Request(url, headers={"User-Agent": "finin-research-prototype/0.1"})
            print(f"Downloading {filename} from pinned revision {REVISION[:12]}...")
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    with temporary.open("wb") as output:
                        shutil.copyfileobj(response, output, length=1024 * 1024)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)

        files[filename] = {
            "url": url,
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        }
        if previous_manifest and filename in previous_manifest.get("files", {}):
            if files[filename]["sha256"] != previous_manifest["files"][filename]["sha256"]:
                raise SystemExit(f"Existing file differs from the recorded checksum: {destination}")
        print(f"  {filename}: {files[filename]['bytes']:,} bytes")

    manifest = {
        "dataset": DATASET,
        "revision": REVISION,
        "downloaded_at_utc": (
            previous_manifest["downloaded_at_utc"]
            if previous_manifest
            else datetime.now(timezone.utc).isoformat(timespec="seconds")
        ),
        "dataset_card": f"https://huggingface.co/datasets/{DATASET}",
        "files": files,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest: {MANIFEST_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

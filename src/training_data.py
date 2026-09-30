"""Join prepared dates to immutable frozen feature arrays for PyTorch."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.nifty import sha256_file
from src.features.cache import source_signature, selected_texts


class PreparedDataset(Dataset):
    def __init__(self, root: Path, split: str, use_text: bool = True):
        self.rows = [json.loads(line) for line in (root / "data" / "processed" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        self.use_text = use_text
        self.description = np.zeros(384, dtype=np.float32)
        self.index: dict[str, int] = {}
        if use_text:
            directory = root / "artifacts" / "features"
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            expected = source_signature(root, selected_texts(root))
            if manifest["signature"] != expected:
                raise ValueError("Feature cache does not match prepared data")
            for filename, expected_hash in (
                ("headline_index.json", manifest["headline_index_sha256"]),
                ("text.npy", manifest["states"]["text"]["sha256"]),
                ("sentiment.npy", manifest["states"]["sentiment"]["sha256"]),
                ("market_description.npy", manifest["market_description_sha256"]),
            ):
                if sha256_file(directory / filename) != expected_hash:
                    raise ValueError(f"Feature cache file changed: {filename}")
            self.index = {headline_id: i for i, headline_id in enumerate(json.loads((directory / "headline_index.json").read_text(encoding="utf-8")))}
            self.text = np.load(directory / "text.npy", mmap_mode="r")
            self.sentiment = np.load(directory / "sentiment.npy", mmap_mode="r")
            self.description = np.load(directory / "market_description.npy").astype(np.float32)
            if len(self.index) != self.text.shape[0] or self.sentiment.shape != (len(self.index), 3):
                raise ValueError("Feature cache shape mismatch")
        else:
            self.text = None
            self.sentiment = None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        vectors = np.zeros((16, 384), dtype=np.float32)
        scores = np.zeros((16, 3), dtype=np.float32)
        mask = np.zeros(16, dtype=bool)
        if self.use_text:
            indices = [self.index[headline["id"]] for headline in row["headlines"]]
            count = len(indices)
            vectors[:count] = self.text[indices]
            scores[:count] = self.sentiment[indices]
            mask[:count] = True
        return {
            "market_features": torch.tensor(row["market_features"], dtype=torch.float32),
            "market_description": torch.from_numpy(self.description.copy()),
            "headline_vectors": torch.from_numpy(vectors),
            "sentiment": torch.from_numpy(scores),
            "mask": torch.from_numpy(mask),
            "target": torch.tensor(row["target_up"], dtype=torch.float32),
            "row_index": index,
        }

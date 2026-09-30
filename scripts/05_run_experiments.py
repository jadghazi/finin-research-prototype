"""Stage 4: run the fixed seven-way comparison and save every prediction."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_fingerprint() -> str:
    paths = [
        ROOT / "data" / "processed" / "manifest.json",
        ROOT / "artifacts" / "features" / "manifest.json",
        ROOT / "src" / "training.py",
        ROOT / "src" / "training_data.py",
        ROOT / "src" / "evaluation.py",
        ROOT / "src" / "models" / "encoders.py",
        ROOT / "src" / "models" / "attention.py",
        ROOT / "src" / "models" / "finin.py",
        ROOT / "src" / "models" / "baselines.py",
        ROOT / "scripts" / "05_run_experiments.py",
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    digest.update(b"epochs30-patience5-batch16-lr0.001-seeds17-42-73")
    return digest.hexdigest()


def main() -> None:
    import torch
    from torch.utils.data import DataLoader

    from src.evaluation import classification_metrics
    from src.training import VARIANTS, predict, train_one
    from src.training_data import PreparedDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    fingerprint = run_fingerprint()
    datasets = {split: PreparedDataset(ROOT, split) for split in ("train", "valid", "test")}
    validation_loader = DataLoader(datasets["valid"], batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(datasets["test"], batch_size=32, shuffle=False, num_workers=0)
    output = ROOT / "artifacts" / "runs"
    output.mkdir(parents=True, exist_ok=True)
    targets = [row["target_up"] for row in datasets["test"].rows]
    validation_targets = [row["target_up"] for row in datasets["valid"].rows]
    runs = []
    baseline_file = output / "always_up.json"
    if baseline_file.exists():
        baseline = json.loads(baseline_file.read_text(encoding="utf-8"))
        if baseline["fingerprint"] != fingerprint:
            raise ValueError("Existing baseline was produced by different code/data")
    else:
        baseline = {
            "variant": "always_up",
            "fingerprint": fingerprint,
            "test_metrics": classification_metrics(targets, [1.0] * len(targets)),
            "validation_metrics": classification_metrics(validation_targets, [1.0] * len(validation_targets)),
            "predictions": [
                {"news_date": row["news_date"], "forecast_date": row["forecast_date"],
                 "target_date": row["target_date"], "target_up": row["target_up"], "probability_up": 1.0}
                for row in datasets["test"].rows
            ],
        }
        baseline_file.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    runs.append(baseline)

    for variant in VARIANTS:
        for seed in (17, 42, 73):
            path = output / f"{variant}_{seed}.json"
            if path.exists():
                record = json.loads(path.read_text(encoding="utf-8"))
                if record["fingerprint"] != fingerprint:
                    raise ValueError(f"Existing run differs from code/data: {path}")
                print(f"Reusing completed run: {variant} seed {seed}", flush=True)
            else:
                checkpoint = output / f"{variant}_{seed}.pt"
                model, training = train_one(variant, seed, datasets, device, checkpoint)
                validation_scores, _ = predict(model, validation_loader, device)
                scores, attention = predict(model, test_loader, device)
                predictions = []
                for row, probability, weights in zip(datasets["test"].rows, scores, attention):
                    top_id = None
                    top_weight = None
                    if weights:
                        top_position = max(range(len(row["headlines"])), key=lambda i: weights[i])
                        top_id = row["headlines"][top_position]["id"]
                        top_weight = weights[top_position]
                    predictions.append({
                        "news_date": row["news_date"], "forecast_date": row["forecast_date"],
                        "target_date": row["target_date"], "target_up": row["target_up"],
                        "probability_up": probability, "top_headline_id": top_id,
                        "top_attention_weight": top_weight,
                    })
                record = {
                    **training,
                    "fingerprint": fingerprint,
                    "device": device,
                    "torch_version": torch.__version__,
                    "validation_metrics": classification_metrics(validation_targets, validation_scores),
                    "test_metrics": classification_metrics(targets, scores),
                    "predictions": predictions,
                }
                path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
                print(f"Completed {variant} seed {seed}: test accuracy {record['test_metrics']['accuracy']:.3f}", flush=True)
                del model
                if device == "cuda":
                    torch.cuda.empty_cache()
            runs.append(record)

    combined = {"fingerprint": fingerprint, "device": device, "runs": runs}
    (output / "all_runs.json").write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    print("All fixed experiments complete. Run scripts/06_export_results.py for meeting outputs.")


if __name__ == "__main__":
    main()

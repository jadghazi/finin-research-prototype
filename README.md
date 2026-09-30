# FININ research prototype

This project will test the central idea of [Wang, Cohen, and Ma (2024)](https://aclanthology.org/2024.findings-emnlp.189/): whether modeling interactions among daily news headlines and weighting those headlines against the market helps predict next-trading-day direction.

This is a **method prototype on public proxy data**, not a numerical replication of the paper's Reuters/TRNA study. The detailed scope and differences are in [REPRODUCTION_PLAN.md](REPRODUCTION_PLAN.md). The data flow and intended code map are in [docs/architecture.md](docs/architecture.md).

## Where we are

**Step 1: inspect candidate data.** The only executable code so far downloads and audits [NIFTY](https://huggingface.co/datasets/raeidsaqur/NIFTY), a public dated-headline dataset. There is no prediction model yet. The audit report is generated at `reports/data_audit.md`. NIFTY is a provisional choice; [docs/dataset_selection.md](docs/dataset_selection.md) records the comparison and what must pass before modeling.

## Reproduce Step 1

Use Python 3.10 or later from a terminal in this folder (VS Code's terminal is fine):

```powershell
python scripts/01_download_nifty.py
python scripts/02_audit_nifty.py
```

The downloader pins one exact dataset revision in its source. It records source URLs, byte sizes, and SHA-256 hashes in `data/nifty_manifest.json`. The audit checks the three JSONL files and writes `reports/data_audit.md`.

The first audit found that all standalone `context` fields contain only a header; market-history rows are embedded in the `conversations` prompts instead. We must inspect those rows and verify SPY prices before defining the binary target. The audit also found a large decline in headlines per day between the training and test periods; this is a real limitation of this proxy dataset.

The NIFTY authors define its three-class label from the move **into** a row's date. FININ's target is the move **after** that date. We will derive the latter from verified prices and will not train on NIFTY's supplied labels.

Raw downloaded headlines are kept in `data/raw/nifty/` and excluded from Git. The source dataset card lists an MIT license, but the headlines originate from news publishers; check redistribution rights before publishing any raw data. The manifest and audit report are safe to keep with the project.

## Planned code map

| File | Role |
| --- | --- |
| `scripts/01_download_nifty.py` | Fetch the pinned headline dataset and record checksums |
| `scripts/02_audit_nifty.py` | Inspect schema, dates, volume, labels, and missing values |
| `scripts/03_prepare_examples.py` | **Planned:** join headlines with SPY prices and create time-safe examples |
| `scripts/04_cache_text_features.py` | **Planned:** extract and save frozen text vectors |
| `src/model.py` | **Planned:** news attention, market attention, prediction head |
| `scripts/05_train.py` | **Planned:** fit baselines and prototype using training/validation dates |
| `scripts/06_evaluate.py` | **Planned:** held-out comparisons and diagnostic plots |

Each script should have one job and print its inputs and outputs. We will implement later stages only after checking the data and target definition.

# FININ research prototype

Goal: reproduce the core architecture of [Wang, Cohen and Ma (2024)](https://aclanthology.org/2024.findings-emnlp.189/) at student scale, run controlled experiments on a 16 GB RAM / GTX 1660 Ti workstation, and present saved results offline on an 8 GB laptop.

**Status: prototype implemented and measured.** Price and news preparation produces 2,107 dated examples. All 32,286 unique selected headlines have frozen text and sentiment features. Seven methods were compared on 317 held-out dates, including 18 trained runs. The model-integrity tests pass. See [held-out results](reports/results.md); the offline meeting walkthrough is generated at `artifacts/meeting/FININ_walkthrough.html`.

## Version 1 scope

Use NIFTY financial headlines, separately snapshotted SPY prices and frozen financial TinyBERT sentiment probabilities. BGE-small supplies the frozen text vectors; the paper also evaluated BGE as a text encoder. Preserve FININ's text/numeric fusion, news self-attention, market-query attention and predictor. Cap news at 16 headlines/example with one input day. Compare simple baselines and ablations on identical held-out dates.

Use the previous trading day's news at the forecast day's close and predict the next session's direction. This is an explicit timing adjustment for missing publication timestamps. The prototype's classification results cannot be directly compared with the paper's original Reuters-data scores. Its full model was close to an always-up prediction on this test period; balanced accuracy shows no convincing directional benefit from attention.

## Read in this order

1. [Reproduction plan](REPRODUCTION_PLAN.md): fixed data/model choices, experiment, compute and work stages.
2. [Dataset decision](docs/dataset_selection.md): every required field and its source.
3. [Verification evidence](reports/plan_verification.md): measured checks, corrections and unknowns.
4. [Architecture and code map](docs/architecture.md): diagrams and module-to-paper mapping.
5. [Raw audit](reports/data_audit.md): news-file counts and quality checks.
6. [Runbook](docs/runbook.md): commands, outputs and what to explain at each stage.
7. [Held-out results](reports/results.md): measured comparison and interpretation.

## Important correction

All 2,111 supplied NIFTY returns match **next-trading-session** SPY returns. Earlier documentation claiming current-day alignment was incorrect. The three-class labels still differ from FININ's binary target, and the delayed-news protocol requires its own alignment. Targets were calculated explicitly from prices.

Standalone `context` fields contain only a header, but prompts embed historical prices that have now been independently cross-checked. The planned separate price snapshot provides a complete calendar without production prompt parsing.

## Existing audit commands

These inspect the news source; they do not build or train the prototype. Python 3.10 or newer:

```powershell
python scripts/01_download_nifty.py
python scripts/02_audit_nifty.py
```

The downloader pins a revision and records checksums in `data/nifty_manifest.json`. The raw audit verifies them and inspects records. `scripts/03_prepare_data.py` independently compares all 2,111 supplied returns with the frozen SPY price series and stops if alignment fails.

## Portability and version control

Local Git is initialized. No GitHub repository is published. Version code, configuration, documentation, manifests and compact reports. Keep raw data and model/cache files outside a public repository and provide acquisition commands plus a separate private transfer where needed.

The laptop meeting package is in `artifacts/meeting/`: a self-contained HTML report, saved predictions and speaking notes. Copy that folder to the laptop; presentation requires no Python, CUDA or model weights. The raw data, feature arrays and trained checkpoints remain on the workstation and are excluded from Git.

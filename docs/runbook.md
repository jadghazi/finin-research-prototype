# Run the FININ prototype, one stage at a time

This guide is for Windows PowerShell in VS Code, opened at the project root. The stronger workstation prepares features and trains the model. The laptop can show the saved meeting package offline.

## 0. Set up Python

Use Python 3.12. Create a virtual environment once:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.5.1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The official CUDA wheel is approximately 2.4 GB and downloaded too slowly on the initial workstation connection. This version uses the smaller CPU wheel. The workstation's GTX 1660 Ti remains available if a compatible CUDA wheel is installed later. Check the actual runtime with:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`torch.cuda.is_available()` is expected to be false with this wheel. The 100-headline benchmark measured CPU speed before the full cache run.

## 1. Prepare dated examples

```powershell
.\.venv\Scripts\python.exe scripts/01_download_nifty.py
.\.venv\Scripts\python.exe scripts/02_audit_nifty.py
.\.venv\Scripts\python.exe scripts/03_prepare_data.py
```

What happens: NIFTY's downloaded revision is checked by hash; a SPY price snapshot is saved and checked; every source return is compared with the next trading session; and 2,107 time-aligned examples are written under `data/processed/`.

What to show your professor: [the three worked examples](../reports/data_preparation.md). A news date precedes the forecast date, and the target close is a further session in the future. The exact excluded dates and standardization constants are in `data/processed/manifest.json`.

## 2. Benchmark and save frozen text features

```powershell
.\.venv\Scripts\python.exe scripts/04_cache_features.py benchmark
.\.venv\Scripts\python.exe scripts/04_cache_features.py cache
```

The first command downloads the pinned BGE-small and financial TinyBERT files if absent, then runs 100 selected headlines through each frozen model and writes `reports/feature_benchmark.json`, including actual time and any available peak GPU memory. Initial model downloads can take longer than inference. The second caches the full set of unique selected headline vectors and sentiment scores under `artifacts/features/`. It can resume completed batches after interruption. The model revisions and score order are recorded in the cache manifest.

What to explain: BGE-small converts words into vectors; a small finance-trained TinyBERT produces three numeric sentiment inputs. The FININ modules train on these saved features. Reuters sentiment is unavailable in the public dataset, so these scores are a documented substitute. The original RoBERTa/FinBERT plan was changed because their weights and the CUDA runtime required several gigabytes of downloads on a measured slow connection.

## 3. Check model integrity

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests check trading-session alignment, split boundaries, headline selection, padding invariance, normalized attention weights, and gradient flow through both attention components. The tiny model test does not assess forecasting accuracy; held-out results come from the next stage.

## 4. Run the fixed comparisons

```powershell
.\.venv\Scripts\python.exe scripts/05_run_experiments.py
```

This runs always-up and six trainable variants on the same dates, with three seeds each. Checkpoints are selected using validation loss. Completed runs and dated test probabilities go under `artifacts/runs/`; an interrupted set of experiments can resume from saved runs if code and data are unchanged.

What to explain: compare the complete model with prices only, average sentiment, mean pooling, no news self-attention and no sentiment. A weaker result is still a real answer to the question on this sample.

## 5. Make the laptop presentation

```powershell
.\.venv\Scripts\python.exe scripts/06_export_results.py
```

Show `artifacts/meeting/FININ_walkthrough.html`. It is a single offline page with the pipeline, worked dates, comparison chart, metrics and example predictions. `artifacts/meeting/predictions.csv` contains the complete selected run's test predictions, and `artifacts/meeting/talking_points.md` is a short speaking script. The tracked [results report](../reports/results.md) contains the same metrics in text form.

Copy the `artifacts/meeting/` folder to the laptop by USB or another private transfer. It does not require installing Python, CUDA or model weights on the laptop. Open the HTML file locally once before the meeting to confirm it displays correctly.

## Data and interpretation boundaries

The program uses previous-trading-day news because NIFTY does not provide individual publication times. The prototype uses SPY instead of the index and financial TinyBERT scores instead of Reuters/TRNA scores. It measures historical classification performance, not an executable trading strategy. Dataset and model downloads are excluded from Git so a public repository stays small and does not redistribute raw headlines.

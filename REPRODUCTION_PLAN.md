# FININ: reduced research prototype plan

Status: all five stages completed on 2026-09-30 on the workstation. The local offline meeting package is generated; opening its copied file on the 8 GB laptop remains a presentation check. This document records the fixed research protocol and implementation choices.

## Goal and completion criteria

Build a small, understandable reproduction of FININ's core architecture that the student research assistant can train, test, and explain to their professor. Run experiments on the 16 GB RAM / GTX 1660 Ti workstation. Present saved results offline on the 8 GB laptop. Use VS Code, a documented Python project, and version control so another researcher can follow the work.

Success means a working, reproducible experiment with honest results, including negative results. It does not require matching the paper's scores. Preserve the principal model components, reduce the workload, and record data substitutions. Original Reuters data is an optional future improvement, not a prerequisite for this prototype.

The verified outputs are in the [preparation report](reports/data_preparation.md), [feature benchmark](reports/feature_benchmark.json), and [held-out results](reports/results.md). The full model largely predicted up on the test period; attention showed no convincing directional benefit in this prototype.

## Fixed first-experiment decisions

| Component | Decision |
| --- | --- |
| News | Already downloaded NIFTY JSONL, revision `9b8aef736cbaf6a7e9645ac20e8bd9b2344734d2`; use its `news` field |
| Market | SPY, the S&P 500 ETF proxy; Yahoo daily OHLCV and adjusted close, frozen in `data/raw/spy/daily.json` with a tracked checksum manifest |
| Target | `y_d = int(close[next_session(d)] > close[d])` |
| News timing | At forecast day d, use the previous trading day's dated news bucket, a deliberate delay for missing publication times |
| Sentiment | Frozen financial TinyBERT ONNX model (`mikeysharma/finance-sentiment-analysis`); three positive/neutral/negative probabilities cached once |
| Text vectors | Frozen `BAAI/bge-small-en-v1.5`, shared for headlines and market description; cache final first-token vectors |
| Workload | At most 16 unique headlines/example; one market, one input day, one chronological split |
| Model | Shared learned text projection; separate numerical and fusion MLPs; news self-attention; market-query attention; final MLP |
| Evaluation | Held-out classification results, simple baselines and controlled ablations, three fixed seeds |
| Presentation | Offline report, figures, saved predictions, diagrams and documented commands |

These choices define version 1. Extra markets, alternative datasets, longer lookbacks, broad parameter searches and trading backtests are later extensions.

## What has actually been verified

Details are in [the verification report](reports/plan_verification.md).

- Read FININ sections 3-5, Figure 2 and Appendix A/B against the local PDF.
- Compared every NIFTY row with a separately retrieved Yahoo SPY price series: **all 2,111 supplied returns match next-trading-day returns within source rounding precision.** The previous assertion that they describe movement into the row date was incorrect.
- Recovered 2,695 unique historical price rows from the prompts, without conflicting duplicates. All OHLC prices agree with Yahoo within approximately 0.000015 dollars; volumes match exactly.
- Confirmed the separate Yahoo response covers every date needed by the proposed delayed-news experiment, through final target date 2020-09-23, with no missing OHLCV or adjusted-close values.
- Inspected the original planned FinBERT and RoBERTa configurations, then measured the runtime/model download sizes on this connection. The PyTorch CUDA wheel alone is about 2.4 GB and downloaded at about 0.15 MB/s. A cached BGE-small checkpoint and a 55 MB finance-trained sentiment ONNX model were selected as practical substitutes. Both ran successfully on all 32,286 unique selected headlines.
- Confirmed a GTX 1660 Ti with 6,144 MiB VRAM. This measured experiment used CPU PyTorch. The latest 100-headline text and sentiment inference times were 0.67 and 0.14 seconds respectively after loading, with zero truncations in that sample; see the benchmark for details.

NIFTY's original three-class label cannot be used directly: Neutral contains both small positive and small negative returns. Compute binary labels from closing prices. For the delayed-news protocol, align labels to the forecast day rather than copying the source row's return.

## Exact example definition and preparation

Let n be a NIFTY headline date, d the next trading session after n, and q the next session after d. Each example uses headlines dated n and market data through the close of d to predict whether close[q] exceeds close[d].

Example: January 6, 2010 headlines, market features through January 7, and the answer from January 8's close versus January 7's close. This remains a one-session forecast. The input news is deliberately delayed relative to FININ's same-day setup.

The protocol assumes NIFTY's date buckets identify the news date correctly. A delay mitigates missing intraday timestamps; it cannot prove historical availability or repair incorrectly dated records. This is a retrospective classification experiment, not evidence of an executable trading strategy.

Preparation follows one fixed recipe:

1. Read headline strings from `news`. The LLM prompt, assistant response, supplied label and supplied return must never enter model features.
2. Use the actual SPY trading calendar, not calendar arithmetic or adjacent NIFTY rows. Prices must extend beyond the last news date to the required target date.
3. Trim whitespace, remove exact duplicate headlines within each date, and select up to 16 by ascending SHA-256 hash of UTF-8 headline text, using the text to break ties. Preserve original text and selection IDs. This deterministic selection does not inspect targets or presume source order reflects relevance.
4. Use six market channels: open/previous close minus 1, high/previous close minus 1, low/previous close minus 1, close/previous close minus 1, adjusted close/previous adjusted close minus 1, and `log1p(volume)`. Standardize using training data only; retain raw prices for label verification.
5. Use a fixed description: "SPY is an exchange-traded fund tracking the S&P 500, a broad index of large U.S. companies." Omit a current constituent list because historical membership has not been obtained.
6. Keep source train/validation/test membership. Remove training examples whose target date reaches the first validation news date (2017-06-28), and validation examples whose target date reaches the first test news date (2019-02-13). Outcome periods must not cross those split boundaries.

Verified expected counts, before any additional content exclusions:

| Split | Raw news dates | Retained examples | Selected headline slots |
| --- | ---: | ---: | ---: |
| Train | 1,477 | 1,475 | 23,592 |
| Validation | 317 | 315 | 5,040 |
| Test | 317 | 317 | 5,041 |
| Total | 2,111 | 2,107 | 33,673 |

Use identical dates and selected headlines for every model. Document changing news coverage across splits. Preserve an exclusion manifest. Missing prices, changed checksums, conflicting dates or failed target checks must stop preparation visibly; never silently change the dataset.

## Sentiment and frozen text features

Use the finance-trained TinyBERT ONNX model [`mikeysharma/finance-sentiment-analysis`](https://huggingface.co/mikeysharma/finance-sentiment-analysis), revision `4353ecd593de607cbb64147c8e2d21b82d842014`. It is 4 layers, hidden width 312, and its card describes negative/neutral/positive outputs. Its config uses generic label names; the expected order is documented and must pass clear positive/negative control headlines in the benchmark before full extraction. Reorder probabilities to positive/neutral/negative. These are text sentiment proxies for TRNA's instrument-related scores. Test their value with a no-sentiment ablation.

Use the locally cached [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5), revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, for 384-dimensional text vectors. The paper evaluated BGE as one of several frozen text encoders; this compact checkpoint and first-token pooling are explicit version-1 choices. Use at most 128 tokens, record truncation counts, and start with batches of 8. Cache by text, model revision and tokenizer settings.

Only projections, MLPs and attention modules train. No language-model fine-tuning or paid LLM API is needed.

## Paper-to-prototype mapping

Source: Wang, Cohen and Ma, [FININ, Findings of EMNLP 2024](https://aclanthology.org/2024.findings-emnlp.189/).

| Paper component | Version 1 |
| --- | --- |
| S&P 500 and NASDAQ 100, 2003-2018 | SPY with available NIFTY dates, 2010-2020 |
| Over 2.7 million TRNA news items | At most 33,673 selected headline slots |
| Reuters sentiment probabilities | Cached finance TinyBERT probabilities; documented substitution |
| Market descriptions and company names | Fixed generic description; historical constituents omitted |
| Frozen LM and shared text projection, Eqs. 1-2 | Retained with the paper-tested BGE family, compact 384-dimensional checkpoint |
| Separate numerical encoders, Eq. 3 | Retained for market features and sentiment |
| Separate market/news fusion MLPs, Eqs. 4-5 | Retained |
| News self-attention and market-query weighting, Eq. 6 | Retained; one layer/head, width 64, masked padding |
| Weighted sum of refined news vectors | Retained; expose weights as diagnostics |
| Final MLP over recent market/news features, Eq. 7 | Retained with one input day |
| Day-d inputs, target d to d+1 | Same target horizon; news delayed one trading session |
| 1/3/5/10/20-day lookbacks and ten sliding windows | One-day input and one chronological split |
| Accuracy, PnL and Sharpe | Accuracy, balanced accuracy, log loss and confusion matrix |

The prototype supports empirical tests of this reduced implementation on substitute data. Its results cannot establish whether the paper's original numerical gains or market claims reproduce.

## Bounded experiment

Run configuration: width 64, two affine layers per MLP with ReLU between them, one self-attention head, one market-query head, dropout 0.1, batch size 16, Adam learning rate 0.001, binary cross-entropy with logits, at most 30 epochs, early stopping after 5 epochs without validation-loss improvement. Seeds: 17, 42, 73. Classification threshold: 0.5. These are our initial settings, not recovered author defaults or empirically optimized settings.

| Comparison | Question |
| --- | --- |
| Always predict up | What does a simple directional benchmark achieve? |
| Price-only MLP | How much can market features explain? |
| Prices plus mean financial sentiment | Does aggregate sentiment add information? |
| Mean-pooled fused news plus market | Does news help without either attention mechanism? |
| FININ without news self-attention | Do interactions among headlines help? |
| FININ without sentiment inputs | Does substitute sentiment help? |
| Full reduced FININ | How does the complete prototype perform? |

The ablations share frozen vectors, prices, fusion components where applicable, dates, selected news and seeds. The six trainable variants require 18 small fits; always-up is deterministic. Select checkpoints using validation loss. Freeze configuration before test evaluation and do not tune from test results. Report all run scores plus mean and standard deviation across seeds; seed variation is not a confidence interval across future market periods.

Show selected predictions and attention weights as diagnostics, not causal influence. A clear negative result satisfies the research prototype goal.

## Compute and portability

Storing one float32 vector for each of the 32,286 unique selected headlines at width 384 requires about 47.29 MiB. Sentiment adds about 0.37 MiB. Metadata, model weights, framework overhead and activations are additional. The first run used CPU PyTorch because the much larger CUDA wheel could not be fetched in a practical time on this connection. The GTX 1660 Ti remains available for a later CUDA-enabled environment. Actual CPU throughput was checked before full extraction.

The first runtime benchmark processed 100 selected headlines with each frozen encoder, checking output dimensions, sentiment controls, elapsed time and GPU memory when applicable. The small trainable model also ran on CPU. Colab is not required by the design.

Use the project's virtual environment and pinned dependencies. `nvidia-smi` confirms the hardware but does not prove PyTorch CUDA compatibility; the first completed environment uses CPU PyTorch. Save environment versions, revisions, commands, configuration, seeds and run logs.

For the laptop, the self-contained offline HTML has an inline chart, a pipeline diagram, worked dates and example predictions. The meeting folder also has a compact CSV and speaking notes. Showing the completed work does not require laptop training, CUDA or large encoder weights.

## Implementation stages and evidence

| Stage | Work | Completion evidence and explanation |
| --- | --- | --- |
| 1. Prepare data | Snapshot SPY prices; make examples, binary targets and selection/split manifests | Expected counts, date assertions, no missing targets, and three examples showing inputs versus future answer |
| 2. Runtime and features | Benchmark 100 headlines; cache BGE text vectors and financial TinyBERT scores | Measured time/memory, finite 384-vectors, probability/label controls, saved revisions |
| 3. Model and baselines | Implement named modules mapped to paper equations | Tiny-batch learning, padding-invariance, finite loss, normalized attention over real news |
| 4. Fixed comparison | Train variants, select on validation, evaluate held-out data | Saved prediction files, metrics and run metadata; no test-driven tuning |
| 5. Meeting package | Export results, figures and a short walkthrough; complete README | Self-contained local HTML, CSV and speaking notes created; actual laptop check remains to be done |

At each stage, explain inputs, outputs, purpose and connection to the paper. The [architecture document](docs/architecture.md) specifies the intuitive code map. Do not scaffold empty files merely to fill a diagram.

## Limitations and later work

News date provenance is assumed; publication times are unavailable. Coverage changes over time. SPY differs from the index, and TinyBERT sentiment differs from TRNA. The small sentiment model's training-data description is less detailed than FININ's original source. Pretrained models were released during or after parts of the historical sample, so this is a retrospective benchmark rather than a point-in-time deployment simulation. One period and a small test set limit generalization claims.

PnL/Sharpe are deferred: no executable original same-day news cutoff has been established, and Appendix B's risk-free-rate units and correctness-flag return formula need clarification before treating them as a trading rule. Longer lookbacks and sliding windows are later extensions.

If AUB provides an authorized TRNA research export, add a second data adapter and a separately versioned experiment. Access is an opportunity, not a dependency. Do not replace inputs midway through evaluating version 1.

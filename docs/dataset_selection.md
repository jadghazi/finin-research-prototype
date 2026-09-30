# Dataset decision

Decision on 2026-09-30: **NIFTY headlines + verified Yahoo SPY prices + generated financial TinyBERT sentiment for version 1.** Use previous-trading-day news at the forecast day's close. This supplies the principal FININ inputs and supports controlled classification experiments within the available hardware.

This replaces the earlier advice to wait for TRNA or restrict NIFTY to a code demonstration. A reduced implementation can produce meaningful results on documented substitute data. Its results answer the prototype question and are not directly comparable to original-paper scores.

## Corrections from direct checks

- **Target timing:** all 2,111 source returns match next-trading-day SPY returns within rounding precision. Earlier advice asserting movement into the row date was wrong. The source label is still three-class, while FININ requires binary labels.
- **Prices:** prompts contain 2,695 distinct price rows, independently matched to Yahoo OHLCV. Standalone `context` fields contain only a header. Use a separate full price snapshot to avoid parsing prompts in the production pipeline.
- **Audit count:** arbitrary headline lines inflated the earlier reported maximum of 109 price-history rows. Actual maxima are 8 in every split.
- **Sentiment:** NIFTY lacks Reuters sentiment; probabilities from a compact finance-trained TinyBERT are an explicit first-version substitute. Its incremental value was tested with a no-sentiment ablation and was not established on this held-out period. This replaced the larger planned FinBERT after download speed was measured.
- **Timing:** no per-headline publication times are available. Previous-trading-day news reduces the risk, while correct source-date grouping remains an assumption.

See [verification evidence](../reports/plan_verification.md) and [the full recipe](../REPRODUCTION_PLAN.md).

## Required fields

| Field | Source | Status |
| --- | --- | --- |
| News text/date | Pinned NIFTY `news` and `date` | 2,111 rows; no empty-news dates |
| Market OHLCV/adjusted close | Frozen Yahoo SPY daily chart response | 2,701 sessions returned; all required fields present; snapshot checksum tracked |
| Up/not-up target | Compare closes on forecast day and next session | Calculated and checked for all 2,107 retained examples |
| Per-headline sentiment | Frozen `mikeysharma/finance-sentiment-analysis` ONNX model | Positive/negative/neutral control inputs passed; all selected headlines cached |
| Market text | Fixed generic SPY/S&P 500 description | Cached as a static input; historical constituents omitted |
| Text vectors | Frozen, locally cached `BAAI/bge-small-en-v1.5` | 384-dimensional inference benchmark passed; all selected headlines cached |

The exact recipe retains 1,475 train, 315 validation and 317 test examples after boundary purging, capped at 16 unique headlines each. Use the complete trading calendar; never treat adjacent NIFTY rows as necessarily adjacent trading sessions. Never convert Neutral mechanically to zero or feed source answers into features.

## Why this choice

NIFTY is small enough to inspect completely, concerns an S&P 500 proxy, and supplies headline sets suitable for FININ's news interaction mechanism. Its prices and return alignment have now been checked numerically. Sentiment and timing substitutions are specified before modeling.

FNSPID may support a future stock-level study. Its larger hosted files do not inherently make a small experiment impossible: streaming/selective extraction may be possible, but that path is not verified here and is unnecessary for version 1. The Reddit/DJIA and sparsely documented S&P 500 Kaggle candidates offer no verified improvement for this prototype.

## Original data and AUB

TRNA includes historical Reuters headlines and instrument-related sentiment. A smaller authorized subset could make a later experiment closer to the original. Access has not been established.

AUB's [Al Katami Trading Room page](https://www.aub.edu.lb/osb/TradingRoom/Pages/Software-and-Technology.aspx) lists Thomson Reuters Eikon; that alone does not establish historical bulk News Analytics access. At the meeting, ask about a research export with headlines, publication timestamps and positive/neutral/negative scores. This does not block version 1.

## Sources

- [NIFTY](https://huggingface.co/datasets/raeidsaqur/NIFTY), pinned revision and checksums in `data/nifty_manifest.json`.
- [NIFTY paper](https://www.cs.toronto.edu/~raeidsaqur/writings/nifty-dataset_raeid.saqur2024.pdf). Its return formula alone did not establish alignment to the released record date; direct checks resolved that ambiguity.
- [Yahoo SPY history](https://finance.yahoo.com/quote/SPY/history/); exact verification request in the evidence report.
- [Financial TinyBERT](https://huggingface.co/mikeysharma/finance-sentiment-analysis), [BGE-small](https://huggingface.co/BAAI/bge-small-en-v1.5), [FININ paper](https://aclanthology.org/2024.findings-emnlp.189/).

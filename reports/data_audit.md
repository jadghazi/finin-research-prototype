# NIFTY raw data audit

Source: [https://huggingface.co/datasets/raeidsaqur/NIFTY](https://huggingface.co/datasets/raeidsaqur/NIFTY)
Pinned revision: `9b8aef736cbaf6a7e9645ac20e8bd9b2344734d2`
Downloaded (UTC): `2026-09-30T09:41:55+00:00`

This audit describes the downloaded source files only. It does **not** establish that the published labels match FININ's binary target or that all headlines were available at a trading cutoff.

| Split | Rows | First date | Last date | Headlines total | Min/day | Median/day | Max/day |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| train | 1,477 | 2010-01-06 | 2017-06-27 | 411,237 | 11 | 78 | 1627 |
| valid | 317 | 2017-06-28 | 2019-02-12 | 17,034 | 18 | 54 | 93 |
| test | 317 | 2019-02-13 | 2020-09-21 | 8,299 | 7 | 26 | 46 |

## Data quality checks

### train

- Labels: `{"Fall": 307, "Neutral": 788, "Rise": 382}`
- Empty-news dates: 0
- Duplicate headline strings within a date: 18,932
- Context fields with only a header/one line: 1,477
- Missing `pct_change`: 0
- Weekend dates: 0

### valid

- Labels: `{"Fall": 53, "Neutral": 189, "Rise": 75}`
- Empty-news dates: 0
- Duplicate headline strings within a date: 242
- Context fields with only a header/one line: 317
- Missing `pct_change`: 0
- Weekend dates: 0

### test

- Labels: `{"Fall": 73, "Neutral": 143, "Rise": 101}`
- Empty-news dates: 0
- Duplicate headline strings within a date: 1
- Context fields with only a header/one line: 317
- Missing `pct_change`: 0
- Weekend dates: 0

## Interpretation for the next step

- Dates are ordered and non-overlapping across splits (the audit would fail otherwise).
- We must choose and record a headline cap/selection rule before feature extraction.
- We must obtain and validate a separate SPY price series to supply market features and derive the paper's binary next-day target.
- The source has date-level news groupings, so release-time availability remains uncertain.
- Every `context` field contains only a header/one line; it cannot supply the market history described in the dataset card.
- Headline volume falls sharply from training to testing. The experiment must report this coverage shift and should compare models on identical dates.

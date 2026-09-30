# Planning feasibility verification

Initial planning check on 2026-09-30. The historical checks below preceded the completed experiment.

Implementation update, same date: Stage 1 subsequently downloaded and froze the SPY snapshot, prepared 2,107 examples, and passed the date/target tests. The initial RoBERTa/FinBERT choices below were replaced in the active experiment by locally cached BGE-small and a compact finance-trained TinyBERT ONNX sentiment model after observing the CUDA wheel's 2.4 GB size and about 0.15 MB/s transfer rate. See the current [plan](../REPRODUCTION_PLAN.md) and [prepared-data report](data_preparation.md). The original configuration checks remain recorded here as planning history, not as claims about the active feature cache.

Final implementation update, same date: the CPU runtime, both frozen feature models, all 32,286 unique headline features, seven integrity tests, 18 trained runs and offline meeting exports completed. See the [benchmark](feature_benchmark.json) and [held-out results](results.md). The earlier untested-runtime statement at the end of this historical audit no longer describes the completed prototype.

## Local source checks

NIFTY revision: `9b8aef736cbaf6a7e9645ac20e8bd9b2344734d2`. JSONL checksums and URLs are in `data/nifty_manifest.json`.

Read every record. Recognize an embedded price row only if it begins with an ISO date and comma and contains 16 parseable CSV fields. Recovered 2,695 distinct price dates, 2010-01-05 through 2020-09-18, with zero conflicting rows, zero malformed recognized rows, and no embedded price dates on/after their containing example date. Recovered OHLC rows are finite and internally ordered; volumes are positive.

The prior audit counted arbitrary lines after `Context:` as price rows. Some prompts include headlines there, inflating the reported maximum to 109. Correct ranges are train 1-8, validation 4-8, test 5-8.

## Independent Yahoo cross-check

Retrieved into memory:

```text
https://query1.finance.yahoo.com/v8/finance/chart/SPY?period1=1262304000&period2=1600992000&interval=1d
```

The response contained 2,701 sessions, 2010-01-04 through 2020-09-24, with no missing OHLCV or adjusted-close values. SHA-256 of the recorded verification response: `a1d21921855d536c35b1d98b584cf162d0d937a0487e3091a2e08132daf35bc5`. The raw response was not persisted. This verifies current access and consistency; implementation must save and hash its own snapshot. Historical adjusted prices can change with later corporate actions.

For all 2,695 common dates, maximum absolute difference in each OHLC field was 0.000014648437514 dollars. Volumes matched exactly. Adjusted-close values across retrieval vintages were not asserted identical.

The complete response includes six sessions missing from the prompt-derived table: 2010-01-04, 2018-12-31, and 2020-09-21 through 2020-09-24. Therefore, adjacent prompt-price rows cannot safely define adjacent trading sessions. Use the separate complete snapshot.

Compared each source `pct_change` with `close[next_session(n)] / close[n] - 1`, tolerance 0.000051 for four-decimal source rounding:

| Split | Matches / rows |
| --- | ---: |
| Train | 1,477 / 1,477 |
| Validation | 317 / 317 |
| Test | 317 / 317 |

Example: the 2010-01-06 row supplies 0.0042. SPY closes are 113.71 on January 6 and 114.19 on January 7; `114.19 / 113.71 - 1 = 0.00422126`. The return is next-session, contrary to earlier advice. Derive binary targets from prices, not rounded returns or a mechanical mapping of Neutral to zero.

## Delayed-news recipe check

News date n; forecast day d = next trading session; target day q = next session after d. Inputs: news n and prices through d. Target: close[q] > close[d].

Keep source split membership and purge training target dates on/after 2017-06-28 and validation target dates on/after 2019-02-13. Removed news dates: 2017-06-26, 2017-06-27, 2019-02-11, 2019-02-12.

| Split | Retained | First (news, forecast, target) | Last (news, forecast, target) | Headline slots after trim/dedup/cap 16 |
| --- | ---: | --- | --- | ---: |
| Train | 1,475 | 2010-01-06, 2010-01-07, 2010-01-08 | 2017-06-22, 2017-06-23, 2017-06-26 | 23,592 |
| Validation | 315 | 2017-06-28, 2017-06-29, 2017-06-30 | 2019-02-07, 2019-02-08, 2019-02-11 | 5,040 |
| Test | 317 | 2019-02-13, 2019-02-14, 2019-02-15 | 2020-09-21, 2020-09-22, 2020-09-23 | 5,041 |

Counted eligible dates/slots without generating production examples or features. The eventual preparation stage must reproduce these checks. Delayed news reduces risk; it does not prove date provenance or historical availability.

## Model configurations and compute

| Item | Observed |
| --- | --- |
| `ProsusAI/finbert` revision | `4556d13015211d73dccd3fdd39d39232506f3e43` |
| FinBERT label IDs | 0 positive, 1 negative, 2 neutral |
| FinBERT architecture | 12 layers; hidden width 768 |
| `FacebookAI/roberta-base` revision | `e2da8e2f811d1448a5b465c236feacd80ffbac7b` |
| RoBERTa architecture | 12 layers; hidden width 768 |
| GPU from `nvidia-smi` | GTX 1660 Ti; 6,144 MiB VRAM |
| Driver | 555.85 |
| Host RAM | 16 GB stated by user; not independently measured here |

33,673 headline slots x 768 float32 values = 98.65 MiB; three sentiment values per slot add 0.39 MiB. Repeated text can reduce stored cache size. This excludes encoder weights, framework overhead, activations and metadata.

The audit runtime has no `torch` or `transformers`. Inference, dependency compatibility, actual peak memory, runtime, predictive quality and laptop presentation have not been tested. The plan includes a 100-headline benchmark before full extraction. The future implementation is not claimed to already run.

## Paper mapping and code availability

Checked local FININ sections 3-5, Figure 2 and Appendix A/B: binary next-session target; text and sentiment per headline; market text and six numerical fields; frozen LM; shared text projection; separate numerical/fusion MLPs; news self-attention; market-query weighting; weighted news aggregation; MLP predictor.

Official code was not identified in the public paper links/title search for this review. This is a bounded search result, not proof that no implementation exists anywhere.

Primary references: [FININ](https://aclanthology.org/2024.findings-emnlp.189/), [NIFTY](https://huggingface.co/datasets/raeidsaqur/NIFTY), [FinBERT](https://huggingface.co/ProsusAI/finbert), [RoBERTa configuration](https://huggingface.co/FacebookAI/roberta-base/raw/main/config.json).

# FININ reproduction: laptop-sized research prototype

## Research question

Can a small, auditable implementation of FININ's **news-to-news attention plus market-to-news attention** improve next-trading-day S&P 500 direction prediction over simple baselines on a public news dataset?

This is a **method reproduction / prototype**, not a numerical replication of the paper. Different news coverage, sentiment scores, dates, and compute mean its reported Sharpe ratios are not direct targets.

## What the paper actually does

Source: Wang, Cohen, and Ma, *Modeling News Interactions and Influence for Financial Market Prediction*, Findings of EMNLP 2024, pp. 3302-3314. Local copy: `paper/Wang et al. - 2024 - Modeling News Interactions and Influence for Financial Market Prediction.pdf`. Official page: https://aclanthology.org/2024.findings-emnlp.189/

- **Task (section 3):** using data through trading day `d`, predict whether the index close on `d+1` is above the close on `d`. Its 1, 3, 5, 10, and 20 day settings are *input lookback lengths*, not forecast horizons.
- **Inputs (sections 3-4, appendix A):** market price features and a market description; each news item has a headline and Reuters/TRNA sentiment scores.
- **Model (section 4, figure 2):** a frozen pretrained language model supplies text vectors; learned encoders fuse each item's text and numbers; same-day news self-attention contextualizes items; market-query cross-attention weights the news items; a predictor consumes recent market/news vectors.
- **Study (sections 5-6):** S&P 500 and NASDAQ 100, 2003-2018, over 2.7 million news reports. Ten chronological 500-day windows are split 400/50/50 days for train/validation/test and shifted by 391 days. The paper reports accuracy, cumulative PnL, and Sharpe ratio, plus ablations.
- **Compute (section 5.2):** the authors report about 32 hours training on a 2080 Ti or 7.5 hours on an A100. That is for their original setup, not a prediction for our machine.

## Data decision

**Primary public prototype:** NIFTY (https://huggingface.co/datasets/raeidsaqur/NIFTY). Its card reports 2,111 dated SPY examples (2010-2020) with chronological train/validation/test partitions of 1,477/317/317. The `news` field contains newline-separated headlines. One inspected date had 849 headlines, so cap the daily set (start with 16 or 32) using a documented deterministic rule. Save the raw source revision and data checksums. NIFTY provides three-class labels, so **derive our own binary target from a verified SPY daily price series**; do not silently map its labels onto FININ's binary task. Obtain daily OHLCV data from a documented source, cache the exact file, and audit calendar/time-zone joins. Use an end-of-calendar-day `d` information cutoff for next-session predictions because individual headline release times are unavailable. Report this timing limitation and do not interpret attention weights as causal news effects.

**Alternative if the professor wants ticker-level news:** FNSPID (https://huggingface.co/datasets/Zihan1004/FNSPID; https://github.com/Zdong104/FNSPID_Financial_News_Dataset) has item-level stock news, prices, and sentiment, but the full hosted collection is about 29.6 GB. Select a small subset by ticker/date; inspect timestamp quality and missing sentiments before using it. It is a different prediction problem if the target is an individual stock. The dataset card states CC BY-NC 4.0. Do not commit a full data dump to GitHub.

**Original data:** Reuters/TRNA is not included in the project folder or the paper. Ask the professor whether the university has licensed access; an exact data replication depends on that.

## Minimal experiment

1. **Audit data:** daily counts, missing dates, duplicate headlines, missing OHLCV, label alignment, and whether a headline could postdate the prediction cutoff. Preserve a manifest of excluded rows and counts.
2. **Create features:** normalize price changes and volume with parameters fitted on training data only. Extract headline embeddings with one frozen, downloadable encoder, in small batches on the GTX 1660 Ti or CPU. Cache embeddings in a compact format. A lexical sentiment score is a cheap first proxy; pretrained FinBERT probabilities are a later, stronger proxy. Neither is equivalent to TRNA sentiment. Do not fine-tune the language model.
3. **Implement the core:** learned price/text fusion, news self-attention, market-query news attention, and a small MLP classifier. Start with a 1-day lookback and 16-32 headlines/day. Mask padding and record which headlines were selected. Add a 5-day lookback only after the first experiment works. Keep model width small (e.g., 32-64), batch size small, and use early stopping on validation loss.
4. **Compare on identical dates:** always-up, price-only classifier, average daily sentiment, mean-pooled headline model, and FININ-style attention model. A key ablation replaces attention pooling with mean pooling; another removes news self-attention. Do not use the test set for model or headline-selection decisions.
5. **Report:** accuracy plus balanced accuracy and a confusion matrix; uncertainty or variation across a few fixed seeds; optional illustrative PnL/Sharpe only with explicit execution time, transaction-cost assumption, risk-free rate units, and a warning that a close-to-close return is not executable after observing that day's close. Show a few attention weights as **diagnostics**, not proof of causation.

## Development setup

Use a local Python project edited in **VS Code** (Visual Studio Code), with a virtual environment and a few small scripts/modules for data preparation, embedding cache, model, training, and evaluation. One notebook can explore data and present figures, but the reproducible pipeline should run from documented commands. The GTX 1660 Ti machine is the primary compute environment. **Google Colab is optional** if local embedding extraction proves too slow or memory constrained; it is not required for the planned prototype or laptop presentation.

## Meeting deliverable and portability

- A Git repository with the exact environment, data acquisition/preparation scripts, one small configuration file, training/evaluation commands, fixed seeds, and a `README` stating every deviation from the paper.
- A frozen tiny sample for a fast smoke test, a trained checkpoint, cached embeddings/features, and a saved test prediction table. Keep licensed or uncertain-rights raw headlines out of the public repository; provide a fetch/preparation script and use a private transfer for cached data if needed.
- A 3-5 minute **offline walkthrough** on the 8 GB laptop: show the repository, the model diagram/data flow, the completed run log, held-out predictions against baselines, and a few selected headline weights. Running training or inference live is optional. Export static plots and a short results summary so the presentation still works without Python or internet access.
- A one-page result sheet with dataset counts, exact time split, metrics for all baselines/ablations, training runtime, hardware, and limitations. A modest or negative result is still useful if the pipeline is auditable.

## Claims and paper details to handle carefully

- The paper does not specify enough preprocessing, time cutoffs, exact selected hyperparameters, or code to guarantee numerical reproduction from the PDF alone.
- The paper's S&P 500 20-day FININ PnL appears as **0.014** in table 2 and **0.041** in tables 1 and 3; avoid treating that figure as an unquestioned numerical target.
- Appendix B.2 uses a risk-free value of **0.02** in its Sharpe definition without clearly specifying the frequency conversion. State whether any prototype Sharpe uses daily or annualized excess returns.
- Longer lookback results do not by themselves prove delayed market pricing or causal news influence. Re-test these as hypotheses with a controlled 1-day versus 5-day comparison.

## Decision for the professor

Ask whether they value **architectural fidelity on a public proxy dataset** (recommended for the next meeting) or **comparison to the original TRNA study** (requires access to the licensed corpus and more compute). Also ask whether the prototype should remain an index-level SPY task or move to ticker-level prediction if a different news source is chosen.

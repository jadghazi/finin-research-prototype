# A five-minute explanation for the meeting

## 1. The task (about 30 seconds)

The paper asks whether daily financial news and market data can predict whether the next trading session closes higher. Its distinctive idea is to model how headlines relate to each other, then let the current market state decide how much to weight each one.

## 2. What I reproduced (about 60 seconds)

I implemented separate text and numerical encoders, news-to-news attention, market-to-news attention, a weighted news summary, and a final classifier. The text encoders are frozen; this first run used the smaller CPU PyTorch wheel after the much larger CUDA wheel proved impractical to download on this connection.

## 3. Data and timing (about 60 seconds)

I used NIFTY financial headlines and checked their dates and returns against a saved SPY price series. Reuters sentiment was not public, so I generated three probabilities per headline with a compact financial sentiment model and tested whether they helped. Since NIFTY has no per-headline publication times, I used the previous trading day's news. There are 1,475 training, 315 validation, and 317 test examples.

Point to a row in the walkthrough: the news date is earlier than the forecast date, and the answer date is one session after the forecast date. Prices from the answer date never enter model features.

## 4. Results (about 90 seconds)

On the held-out dates, reduced FININ mean accuracy was 58.4% across three seeds. Always-up accuracy was 58.0%; mean pooling reached 58.0%. The full-model runs predicted up on 312–317 of 317 dates. Show the balanced-accuracy chart: it reveals that this prototype did not learn to distinguish down days reliably.

All variants used the same dates and selected headlines. The full model was compared with prices only, average sentiment, mean pooling, no news self-attention, and no sentiment. Describe a gain only if the table actually shows one; a negative result still tests the idea honestly.

## 5. What the result means (about 40 seconds)

This is a small experiment that reproduces the paper's core computation, with documented data substitutions. It does not reproduce the published Reuters-data numbers or prove headline causality or a tradable strategy. If historical TRNA access becomes available, this code can take a second data adapter for a closer study.

## Likely questions

- Why SPY? It tracks the S&P 500 and has a complete daily price series over these headline dates.
- Why generate sentiment? FININ uses a numerical sentiment input for each item, and NIFTY does not supply the Reuters scores.
- Why delay news? We cannot verify intraday publication times in this source.
- Why does the laptop not run training? All experiments and figures are already saved; the laptop only displays the walkthrough.
- Are these the authors' numbers? No. The smaller data, sentiment source, timing, and validation design are all documented changes.

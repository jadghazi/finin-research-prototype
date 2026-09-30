"""Stage 5: turn the fixed experiment into readable offline meeting material."""

from __future__ import annotations

import csv
import html
import json
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


ORDER = (
    ("always_up", "Always up"),
    ("price_only", "Prices only"),
    ("price_sentiment", "Prices + average sentiment"),
    ("mean_pool", "News mean pooling"),
    ("no_self_attention", "Market attention only"),
    ("no_sentiment", "FININ without sentiment"),
    ("full", "Reduced FININ"),
)


def summary(runs: list[dict], variant: str, metric: str) -> tuple[float, float]:
    values = [run["test_metrics"][metric] for run in runs if run["variant"] == variant]
    if not values:
        raise ValueError(f"Missing variant: {variant}")
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def chart_svg(runs: list[dict], metric: str, label: str) -> str:
    width, height = 840, 385
    lines = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(label)} by model">']
    for tick in (0, .25, .5, .75, 1):
        x = 292 + tick * 430
        lines.append(f'<line x1="{x:.0f}" y1="25" x2="{x:.0f}" y2="338" stroke="#e4e9f0"/>')
        lines.append(f'<text x="{x:.0f}" y="362" text-anchor="middle" fill="#627184" font-size="13">{tick:.0%}</text>')
    for index, (variant, name) in enumerate(ORDER):
        y = 36 + index * 43
        value, deviation = summary(runs, variant, metric)
        color = "#1769aa" if variant == "full" else "#81b6ca"
        lines.append(f'<text x="280" y="{y+19}" text-anchor="end" fill="#23354d" font-size="15">{html.escape(name)}</text>')
        lines.append(f'<rect x="292" y="{y+4}" width="{430*value:.1f}" height="22" rx="5" fill="{color}"/>')
        lines.append(f'<text x="{min(300+430*value,780):.1f}" y="{y+21}" fill="#23354d" font-size="14" font-weight="600">{value:.1%}</text>')
    lines.append('</svg>')
    return "".join(lines)


def main() -> None:
    results_path = ROOT / "artifacts" / "runs" / "all_runs.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))
    runs = data["runs"]
    expected = {"always_up": 1, **{name: 3 for name, _ in ORDER if name != "always_up"}}
    for variant, count in expected.items():
        if sum(run["variant"] == variant for run in runs) != count:
            raise ValueError(f"Incomplete experiment for {variant}")
    if len({run["fingerprint"] for run in runs}) != 1:
        raise ValueError("Runs have different code/data fingerprints")

    prepared = json.loads((ROOT / "data" / "processed" / "manifest.json").read_text(encoding="utf-8"))
    report_lines = [
        "# FININ prototype: held-out results",
        "",
        "## Research question",
        "",
        "Does modeling interactions among headlines and weighting them against the market help predict the next trading session's SPY direction, compared with matched simpler models?",
        "",
        "The experiment uses previous-trading-day NIFTY headlines, market data through forecast close, binary next-session direction, at most 16 headlines/example, frozen BGE-small text vectors, and financial TinyBERT sentiment scores.",
        "",
        f"Prepared rows: {prepared['split_summary']['train']['examples']:,} train, {prepared['split_summary']['valid']['examples']:,} validation, {prepared['split_summary']['test']['examples']:,} test. Models are selected by validation loss; test dates are not used for tuning.",
        "",
        "## Comparison on the same 317 test dates",
        "",
        "| Model | Accuracy | Balanced accuracy | Log loss |",
        "| --- | ---: | ---: | ---: |",
    ]
    for variant, name in ORDER:
        measures = [summary(runs, variant, metric) for metric in ("accuracy", "balanced_accuracy", "log_loss")]
        formatted = [f"{value:.3f}" + (f" ± {sd:.3f}" if variant != "always_up" else "") for value, sd in measures]
        report_lines.append(f"| {name} | {' | '.join(formatted)} |")
    report_lines.extend([
        "",
        "Six trainable variants each ran with seeds 17, 42 and 73. Variation across seeds describes training stability on these particular dates; it is not a confidence interval over future market periods.",
        "",
        "## Reading the result",
        "",
    ])
    full_acc = summary(runs, "full", "accuracy")[0]
    simple_acc = summary(runs, "mean_pool", "accuracy")[0]
    always_acc = summary(runs, "always_up", "accuracy")[0]
    full_runs = [run for run in runs if run["variant"] == "full"]
    predicted_up_counts = [sum(row["probability_up"] >= 0.5 for row in run["predictions"]) for run in full_runs]
    test_size = prepared["split_summary"]["test"]["examples"]
    observed_up = prepared["split_summary"]["test"]["up_targets"]
    finding = (
        f"The full model's mean accuracy is {full_acc:.3f}; mean pooling is {simple_acc:.3f}; "
        f"always predicting up is {always_acc:.3f}. The full-model seeds predicted up on "
        f"{min(predicted_up_counts)}–{max(predicted_up_counts)} of {test_size} dates, while {observed_up} "
        "dates actually rose. Their near-0.5 balanced accuracy shows that most of the apparent "
        "accuracy comes from the up-day majority. This experiment does not provide evidence that "
        "news attention improved directional classification over the simpler alternatives. It "
        "does not reproduce the paper's reported scores."
    )
    report_lines.append(finding)
    report_lines.extend([
        "",
        "Accuracy is the share of correct directions. Balanced accuracy averages recall for up and down days, so it reveals models that mostly predict one class. Log loss also checks whether predicted probabilities are sensible; lower is better. The always-up baseline assigns a probability of 1.0, so its log loss sharply penalizes every down day; use its accuracy and balanced accuracy for the directional comparison.",
        "",
        "## Limitations",
        "",
        "NIFTY lacks individual publication times, so news is delayed one trading day. SPY substitutes for the S&P 500 index. Financial TinyBERT sentiment substitutes for Reuters instrument-related scores. Frozen text models released during or after some historical dates are used for a retrospective benchmark; attention weights are model diagnostics, not causal influence.",
        "",
        "Complete per-seed results, learning histories, dated predictions and model checkpoints are saved in the ignored `artifacts/runs/` folder. The offline walkthrough and compact CSV are in `artifacts/meeting/`.",
        "",
    ])
    report_path = ROOT / "reports" / "results.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    representative = min(
        (run for run in runs if run["variant"] == "full"),
        key=lambda run: run["best_validation_loss"],
    )
    meeting = ROOT / "artifacts" / "meeting"
    meeting.mkdir(parents=True, exist_ok=True)
    with (meeting / "predictions.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["news_date", "forecast_date", "target_date", "target_up", "probability_up", "top_headline_id", "top_attention_weight"])
        writer.writeheader()
        writer.writerows(representative["predictions"])

    example_rows = []
    for split in ("train", "valid", "test"):
        with (ROOT / "data" / "processed" / f"{split}.jsonl").open(encoding="utf-8") as source:
            row = json.loads(next(source))
        example_rows.append(
            f"<tr><td>{row['news_date']}</td><td>{row['forecast_date']}</td><td>{row['target_date']}</td>"
            f"<td>${row['close_at_forecast']:.2f} → ${row['close_at_target']:.2f}</td>"
            f"<td><span class='badge'>{'Up' if row['target_up'] else 'Down / flat'}</span></td></tr>"
        )
    metric_rows = []
    for variant, name in ORDER:
        accuracy, accuracy_sd = summary(runs, variant, "accuracy")
        balanced, balanced_sd = summary(runs, variant, "balanced_accuracy")
        log_loss, loss_sd = summary(runs, variant, "log_loss")
        mark = " class='emphasis'" if variant == "full" else ""
        metric_rows.append(
            f"<tr{mark}><th scope='row'>{html.escape(name)}</th><td>{accuracy:.1%}"
            f"{' ± '+format(accuracy_sd,'.1%') if variant != 'always_up' else ''}</td>"
            f"<td>{balanced:.1%}{' ± '+format(balanced_sd,'.1%') if variant != 'always_up' else ''}</td>"
            f"<td>{log_loss:.3f}{' ± '+format(loss_sd,'.3f') if variant != 'always_up' else ''}</td></tr>"
        )
    prediction_rows = []
    for row in representative["predictions"][:6]:
        prediction_rows.append(
            f"<tr><td>{row['forecast_date']}</td><td>{row['target_date']}</td>"
            f"<td>{row['probability_up']:.1%}</td>"
            f"<td>{'Up' if row['probability_up'] >= .5 else 'Down / flat'}</td>"
            f"<td>{'Up' if row['target_up'] else 'Down / flat'}</td></tr>"
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>FININ research prototype · meeting walkthrough</title>
<style>
:root{{--ink:#17293d;--muted:#607183;--blue:#1769aa;--teal:#81b6ca;--line:#dce5ec;--bg:#f4f7fa}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,Segoe UI,Arial,sans-serif}}
header{{background:linear-gradient(115deg,#0d2946,#176c8f);color:white;padding:45px max(30px,calc((100vw - 1040px)/2));}}
header small{{font-weight:700;letter-spacing:.11em;text-transform:uppercase;color:#bce5eb}}h1{{font-size:clamp(2rem,4vw,3.2rem);line-height:1.1;margin:10px 0}}header p{{max-width:750px;color:#e1f1f4;font-size:1.12rem}}
main{{max-width:1100px;margin:28px auto;padding:0 24px 42px}}section{{background:white;border:1px solid var(--line);border-radius:16px;padding:28px 32px;margin:20px 0;box-shadow:0 8px 24px rgba(20,42,60,.04)}}
h2{{font-size:1.55rem;margin:0 0 8px}}h3{{font-size:1.05rem;margin:18px 0 8px}}p{{margin:8px 0 16px}}.muted{{color:var(--muted)}}
.stats{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:20px 0}}.stat{{background:#eaf3f7;border-radius:11px;padding:15px}}.stat strong{{display:block;font-size:1.55rem;color:#145f87}}.stat span{{font-size:.85rem;color:#4d6679}}
.flow{{display:flex;align-items:stretch;flex-wrap:wrap;gap:9px;margin:22px 0}}.node{{background:#eff5f9;border:1px solid #b8d1df;border-radius:11px;padding:14px;flex:1 1 150px;text-align:center;font-weight:650}}.arrow{{display:flex;align-items:center;font-size:1.5rem;color:var(--blue)}}
table{{width:100%;border-collapse:collapse;font-size:.95rem}}th,td{{padding:11px 10px;border-bottom:1px solid var(--line);text-align:left}}thead th{{font-size:.82rem;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}}.emphasis{{background:#eaf3fa;font-weight:700}}.scroll{{overflow-x:auto}}.badge{{display:inline-block;background:#d9ebf5;color:#175777;border-radius:12px;padding:3px 9px;font-weight:650}}
.callout{{background:#eaf3fa;border-left:4px solid var(--blue);padding:14px 18px;border-radius:5px;margin:18px 0}}.two{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}.two > div{{border:1px solid var(--line);border-radius:10px;padding:16px}}svg{{width:100%;height:auto}}footer{{color:var(--muted);text-align:center;padding:12px}}@media(max-width:700px){{.stats,.two{{grid-template-columns:1fr 1fr}}section{{padding:20px}}}}@media print{{body{{background:white}}header{{-webkit-print-color-adjust:exact;print-color-adjust:exact}}section{{box-shadow:none;break-inside:avoid}}}}
</style></head><body>
<header><small>Research assistant · meeting walkthrough</small><h1>FININ, at student scale</h1><p>A working prototype of the paper's news interaction model, tested on SPY and public financial headlines. This page is self-contained and opens offline on an 8 GB laptop.</p></header>
<main>
<section><h2>1 · The question</h2><p>Can a model that lets headlines interact, then weighs them against the market, predict the next trading session's direction better than simpler methods on the same dates?</p>
<div class="stats"><div class="stat"><strong>{sum(prepared['split_summary'][part]['examples'] for part in ('train', 'valid', 'test')):,}</strong><span>usable dated examples</span></div><div class="stat"><strong>16</strong><span>maximum headlines per example</span></div><div class="stat"><strong>{test_size}</strong><span>held-out test dates</span></div><div class="stat"><strong>18</strong><span>trained runs across 3 seeds</span></div></div>
<p class="muted">The original paper used millions of Reuters news items. This study uses NIFTY headlines, SPY prices, and finance-trained TinyBERT sentiment as a documented smaller experiment.</p></section>
<section><h2>2 · What the model sees</h2><div class="flow"><div class="node">Previous trading day's headlines<br><small>text + financial sentiment</small></div><span class="arrow">→</span><div class="node">News headlines interact<br><small>self-attention</small></div><span class="arrow">→</span><div class="node">Market weighs news<br><small>cross-attention</small></div><span class="arrow">→</span><div class="node">Tomorrow up?<br><small>probability from an MLP</small></div></div>
<p>The market branch also encodes the SPY price features and a fixed market description. Headline and market text use the same frozen BGE-small encoder and learned projection.</p>
<div class="scroll"><table><thead><tr><th>News date</th><th>Forecast date</th><th>Answer date</th><th>Closing prices</th><th>Answer</th></tr></thead><tbody>{''.join(example_rows)}</tbody></table></div>
<div class="callout"><strong>How to explain the dates:</strong> For January 6 headlines, the model uses market data through January 7's close and predicts whether January 8 closes above January 7. The one-day news delay addresses missing publication times.</div></section>
<section><h2>3 · What we tested</h2><p>Every method uses the same 317 test dates. Six trainable variants run with seeds 17, 42 and 73. Model checkpoints were chosen using validation loss.</p>
{chart_svg(runs,'balanced_accuracy','Balanced accuracy')}
<h3>Exact held-out metrics</h3><div class="scroll"><table><thead><tr><th>Method</th><th>Accuracy ↑</th><th>Balanced accuracy ↑</th><th>Log loss ↓</th></tr></thead><tbody>{''.join(metric_rows)}</tbody></table></div>
<div class="callout"><strong>Result to say aloud:</strong> The full model predicted up on {min(predicted_up_counts)}–{max(predicted_up_counts)} of {test_size} test dates across its three seeds. Only {observed_up} dates actually rose. Balanced accuracy near 50% means this run gives no convincing evidence that headline attention improved direction predictions. This is a useful measured outcome of the prototype.</div>
<p class="muted">Balanced accuracy averages success on up and down days. Log loss also assesses the probability estimates. ± is variation across training seeds, not uncertainty across future years.</p></section>
<section><h2>4 · A few saved predictions</h2><p>The selected full-model run below had the lowest validation loss among its three seeds. These are its first six test predictions; the complete CSV is saved beside this report.</p>
<div class="scroll"><table><thead><tr><th>Forecast</th><th>Answer date</th><th>Probability up</th><th>Predicted</th><th>Observed</th></tr></thead><tbody>{''.join(prediction_rows)}</tbody></table></div>
<p class="muted">Predictions over 50% are labeled up. Attention weights can be inspected in the saved run, but they do not prove that a headline caused a market move.</p></section>
<section><h2>5 · What this does and does not show</h2><div class="two"><div><h3>Reproduced from the paper</h3><p>Separate text and numerical encoders, news self-attention, market-query news weighting, a weighted news summary, and a final direction predictor.</p></div><div><h3>Changed for this study</h3><p>SPY replaces the index, NIFTY replaces Reuters, BGE-small and a compact financial sentiment model supply frozen features, headline count is capped, news is delayed one trading day, and one chronological split replaces ten windows.</p></div></div>
<p>The measured results answer this prototype's question on this historical period. They do not numerically reproduce the Reuters study or establish a tradable strategy.</p></section>
</main><footer>FININ research prototype · saved offline report · code and preparation steps are documented in README.md</footer>
</body></html>"""
    (meeting / "FININ_walkthrough.html").write_text(document, encoding="utf-8")
    talking_points = [
        "# A five-minute explanation for the meeting",
        "",
        "## 1. The task (about 30 seconds)",
        "",
        "The paper asks whether daily financial news and market data can predict whether the next trading session closes higher. Its distinctive idea is to model how headlines relate to each other, then let the current market state decide how much to weight each one.",
        "",
        "## 2. What I reproduced (about 60 seconds)",
        "",
        "I implemented separate text and numerical encoders, news-to-news attention, market-to-news attention, a weighted news summary, and a final classifier. The text encoders are frozen; this first run used the smaller CPU PyTorch wheel after the much larger CUDA wheel proved impractical to download on this connection.",
        "",
        "## 3. Data and timing (about 60 seconds)",
        "",
        "I used NIFTY financial headlines and checked their dates and returns against a saved SPY price series. Reuters sentiment was not public, so I generated three probabilities per headline with a compact financial sentiment model and tested whether they helped. Since NIFTY has no per-headline publication times, I used the previous trading day's news. There are 1,475 training, 315 validation, and 317 test examples.",
        "",
        "Point to a row in the walkthrough: the news date is earlier than the forecast date, and the answer date is one session after the forecast date. Prices from the answer date never enter model features.",
        "",
        "## 4. Results (about 90 seconds)",
        "",
        f"On the held-out dates, reduced FININ mean accuracy was {full_acc:.1%} across three seeds. Always-up accuracy was {always_acc:.1%}; mean pooling reached {simple_acc:.1%}. The full-model runs predicted up on {min(predicted_up_counts)}–{max(predicted_up_counts)} of {test_size} dates. Show the balanced-accuracy chart: it reveals that this prototype did not learn to distinguish down days reliably.",
        "",
        "All variants used the same dates and selected headlines. The full model was compared with prices only, average sentiment, mean pooling, no news self-attention, and no sentiment. Describe a gain only if the table actually shows one; a negative result still tests the idea honestly.",
        "",
        "## 5. What the result means (about 40 seconds)",
        "",
        "This is a small experiment that reproduces the paper's core computation, with documented data substitutions. It does not reproduce the published Reuters-data numbers or prove headline causality or a tradable strategy. If historical TRNA access becomes available, this code can take a second data adapter for a closer study.",
        "",
        "## Likely questions",
        "",
        "- Why SPY? It tracks the S&P 500 and has a complete daily price series over these headline dates.",
        "- Why generate sentiment? FININ uses a numerical sentiment input for each item, and NIFTY does not supply the Reuters scores.",
        "- Why delay news? We cannot verify intraday publication times in this source.",
        "- Why does the laptop not run training? All experiments and figures are already saved; the laptop only displays the walkthrough.",
        "- Are these the authors' numbers? No. The smaller data, sentiment source, timing, and validation design are all documented changes.",
        "",
    ]
    (meeting / "talking_points.md").write_text("\n".join(talking_points), encoding="utf-8")
    print(f"Saved {report_path.relative_to(ROOT)}")
    print(f"Saved {meeting.relative_to(ROOT) / 'FININ_walkthrough.html'}")


if __name__ == "__main__":
    main()

# FININ professor meeting guide

Start with `output/pdf/FININ_meeting_brief.pdf`: a two-page visual brief for the meeting. Page 1 explains the paper's pipeline, the reproduced components and essential data differences. Page 2 shows the code skeleton, a simple timing example, verification and the main result. Its editable LaTeX source has the same filename with `.tex` extension.

The five-page `output/pdf/FININ_progress_report.pdf` provides optional background. Both PDFs need no internet, Python or model weights to present. Copy the brief to the meeting laptop and check that it opens before the meeting.

## Optional walkthrough of the detailed five-page report

1. **Page 1, contribution (60 seconds).** FININ relates headlines to each other, then uses the market state to weight the resulting representations. Trace the market and news branches in the two diagrams. We retain these mechanisms in a smaller implementation.
2. **Page 2, data and timing (60 seconds).** Explain the Reuters-to-NIFTY substitution, SPY proxy and generated sentiment. Follow the January 6/7/8 example. The outcome remains one session ahead, while the news is intentionally delayed because publication times are missing.
3. **Page 3, findings (90 seconds).** The full model has 58.36% accuracy, but always-up already has 58.04%. Balanced accuracy is only 50.41%; two seeds predict up every day. A constant probability estimated on training labels nearly matches log loss. We have a working prototype, without evidence of a news-attention gain in this experiment.
4. **Page 4, evidence (45 seconds).** The seven integrity tests pass; all 2,107 examples reconstruct and all 18 checkpoints replay. A tiny training batch can be fit. Point to the module responsibilities rather than opening code.
5. **Page 5, next decision (45 seconds).** Ask whether the priority is accessible-data validation with longer lookbacks or closer replication with institutional news access. Freeze current results before new experiments.

## Likely questions

- **Is this an exact replication?** No: it is an architectural reproduction at student scale. Data, text checkpoint, sentiment semantics, news timing and temporal evaluation differ.
- **Why BGE when the paper uses RoBERTa?** BGE was among the paper's candidates. A compact locally available checkpoint made the first CPU experiment practical. The chosen BGE-small checkpoint is not established as the authors' BGE configuration.
- **Why not change the model to raise accuracy now?** The test set has already been inspected. New model selection belongs on training/validation data with a fresh or explicitly declared temporal evaluation.
- **Could the code be broken because it predicts one class?** Data and checkpoint replay, masking/gradient tests and tiny-batch fitting all pass. They reduce implementation concerns but cannot prove every modeling choice is optimal. Class collapse still needs training/validation diagnosis.
- **Does high attention mean a headline moved the market?** No. It is a within-example model weight. No causal experiment was performed.
- **Does the paper attend across 20 days at once?** Its news self-attention is within each day; the final predictor combines daily representations over the chosen lookback.
- **Why no trading results?** Missing precise publication cutoffs and unresolved execution/metric conventions make a trading claim premature. Appendix B's correctness-based sign flag and risk-free-rate units should be clarified before financial replication.

## Verification and rebuild

From the project root in PowerShell:

```powershell
& .venv/Scripts/python.exe -m unittest discover -s tests -v
& .venv/Scripts/python.exe scripts/08_verify_prototype.py
& .venv/Scripts/python.exe scripts/06_export_results.py
& .venv/Scripts/python.exe scripts/09_build_progress_report.py --tectonic tmp/tools/tectonic/tectonic.exe
```

The audit writes `reports/prototype_audit.json` and leaves the original data, features and experiments unchanged. It requires the workstation's local artifacts. It checks source returns, reconstruction and scaling, cache checksums and shapes, checkpoint predictions, validation selection, attention diagnostics and a tiny-batch optimization smoke check.

The report builder refreshes the six trained result rows from run JSON and compiles the source with Tectonic. A local Tectonic 0.17.0 executable was downloaded from the official project release into ignored `tmp/tools/`; the first compilation fetches standard TeX packages. Alternatively, upload the single `.tex` file to Overleaf and compile with pdfLaTeX. The source includes its diagrams and references and needs no external images or bibliography files.

## Review outcome

No blocking defect was found in the implemented reduced model. The audit and report are new. The existing export gained a training-prior probability comparison; original trained checkpoints and test predictions were preserved. The architecture documentation now correctly distinguishes gradient tests from the newly measured tiny-batch fitting check. This audit establishes local replayability, not a clean-machine reinstall or complete point-in-time data provenance.

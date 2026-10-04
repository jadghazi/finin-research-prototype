# FININ professor meeting package

Open the [two-page visual brief](output/pdf/FININ_meeting_brief.pdf) first. It compares the paper's pipeline with the prototype, the key data substitutions, and the measured result. The PDF opens offline on the meeting laptop.

If you need detail during questions, use the [five-page progress report](output/pdf/FININ_progress_report.pdf), the [speaking notes](artifacts/meeting/talking_points.md), or the [offline walkthrough](artifacts/meeting/FININ_walkthrough.html). The [original paper](paper/Wang%20et%20al.%20-%202024%20-%20Modeling%20News%20Interactions%20and%20Influence%20for%20Financial%20Market%20Prediction.pdf) is included for reference. The [saved prediction CSV](artifacts/meeting/predictions.csv) is optional supporting evidence.

The project code, LaTeX sources, reproducibility notes and compact audit are also in this repository. The meeting materials need no Python installation, internet connection, or model weights. Raw datasets, frozen feature arrays and trained checkpoints are intentionally excluded; the commands to reacquire or rebuild them are in [the runbook](docs/runbook.md).

**Message for the meeting:** The reduced FININ architecture is implemented and verified. Its mean test accuracy is 58.4% versus 58.0% for always predicting up, so this first experiment does not show a clear forecasting benefit.

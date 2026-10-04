# FININ research prototype

A small reproduction of the main architecture in [Wang, Cohen and Ma (2024)](https://aclanthology.org/2024.findings-emnlp.189/): headline self-attention, market-conditioned headline weighting, and next-session direction prediction. This version uses NIFTY headlines, SPY prices, frozen BGE-small text features, and generated financial sentiment. It is an architectural prototype, not a numerical replication of the Reuters study.

## For the professor meeting

Open the [two-page visual brief](output/pdf/FININ_meeting_brief.pdf) first. It compares the paper and prototype pipelines, datasets, timing, and measured result. The [detailed progress report](output/pdf/FININ_progress_report.pdf), [offline walkthrough](artifacts/meeting/FININ_walkthrough.html), [saved predictions](artifacts/meeting/predictions.csv), and [paper PDF](paper/Wang%20et%20al.%20-%202024%20-%20Modeling%20News%20Interactions%20and%20Influence%20for%20Financial%20Market%20Prediction.pdf) are also included. The PDFs have editable LaTeX sources beside them. These meeting files open without Python or internet access.

## Code and reproducibility

- [Reproduction plan](REPRODUCTION_PLAN.md): data, model, and evaluation choices.
- [Architecture and code map](docs/architecture.md): pipeline diagram and module responsibilities.
- [Runbook](docs/runbook.md): setup, preparation, training, evaluation, and report commands.
- [Prepared-data report](reports/data_preparation.md), [source audit](reports/data_audit.md), and [held-out results](reports/results.md): generated checks and measurements.
- [Prototype audit](reports/prototype_audit.json): compact reconstruction and checkpoint checks.

The prototype prepared 2,107 dated examples and evaluated seven methods on 317 held-out dates. Across three seeds, reduced FININ reached 58.4% accuracy versus 58.0% for always predicting up; balanced accuracy was 50.4%. It mostly predicted up, so this experiment has not shown a clear directional benefit from news attention.

Raw datasets, feature arrays, and checkpoints are excluded from Git. Use the runbook to rebuild them on a training machine; the saved meeting materials and code are available in this repository.

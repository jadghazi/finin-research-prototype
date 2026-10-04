"""Refresh the report's measured table and compile its editable LaTeX source."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    'price_only': 'Prices only',
    'price_sentiment': 'Prices + mean sentiment',
    'mean_pool': 'Mean-pooled fused news',
    'no_self_attention': 'No news self-attention',
    'no_sentiment': 'No sentiment input',
    'full': r'\textbf{Reduced FININ}',
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tectonic', default=shutil.which('tectonic'))
    args = parser.parse_args()
    audit = json.loads((ROOT / 'reports/prototype_audit.json').read_text())
    data = json.loads((ROOT / 'artifacts/runs/all_runs.json').read_text())
    assert audit['status'] == 'PASS' and audit['original_fingerprint'] == data['fingerprint']
    path = ROOT / 'output/pdf/FININ_progress_report.tex'
    source = path.read_text(encoding='utf8')
    for variant, label in LABELS.items():
        runs = [r for r in data['runs'] if r['variant'] == variant]
        assert len(runs) == 3
        cells = []
        for metric in ('accuracy', 'balanced_accuracy', 'log_loss'):
            values = [r['test_metrics'][metric] for r in runs]
            factor, digits = (1, 4) if metric == 'log_loss' else (100, 2)
            cell = f'{statistics.mean(values)*factor:.{digits}f} $\\pm$ {statistics.stdev(values)*factor:.{digits}f}'
            cells.append(r'\textbf{' + cell + '}' if variant == 'full' else cell)
        new = label + ' & ' + ' & '.join(cells) + r'\\'
        source, count = re.subn('^' + re.escape(label) + r' & .*$', lambda _: new, source, flags=re.M)
        assert count == 1, (variant, count)
    path.write_text(source, encoding='utf8')
    if not args.tectonic:
        raise SystemExit('Metrics refreshed. Compile the .tex with Tectonic or upload it to Overleaf.')
    subprocess.run([str(Path(args.tectonic).resolve()), '-k', '--outdir', str(path.parent), str(path)], check=True)


if __name__ == '__main__':
    main()

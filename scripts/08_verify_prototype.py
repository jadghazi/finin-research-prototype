"""Offline, non-destructive audit of data, checkpoints and saved FININ results.

Adds a training-prevalence probability reference after the original experiment;
it does not retrain, select a model, or change any original experiment artifacts.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.examples import apply_standardizer, build_examples, fit_standardizer
from src.data.nifty import load_nifty, sha256_file
from src.data.prices import load_prices
from src.evaluation import classification_metrics
from src.training import VARIANTS, evaluate_loss, make_model, predict
from src.training_data import PreparedDataset


def main() -> None:
    torch.set_num_threads(4)
    news = load_nifty(ROOT)  # verifies source checksums
    sessions, prices = load_prices(ROOT)
    positions = {day: i for i, day in enumerate(sessions)}
    source_count = 0
    for rows in news.values():
        for row in rows:
            day = row['date']
            observed = prices[sessions[positions[day] + 1]]['close'] / prices[day]['close'] - 1
            assert abs(observed - float(row['pct_change'])) <= 0.000051
            source_count += 1
    rebuilt, exclusions = build_examples(news, sessions, prices)
    mean, scale = fit_standardizer(rebuilt['train'])
    apply_standardizer(rebuilt, mean, scale)
    manifest = json.loads((ROOT / 'data/processed/manifest.json').read_text())
    assert exclusions == manifest['excluded']
    assert mean == manifest['market_train_mean'] and scale == manifest['market_train_scale']
    for split, rows in rebuilt.items():
        path = ROOT / f'data/processed/{split}.jsonl'
        assert sha256_file(path) == manifest['split_summary'][split]['sha256']
        assert rows == [json.loads(line) for line in path.read_text(encoding='utf8').splitlines()]
    for left, right in [('train', 'valid'), ('valid', 'test')]:
        assert max(r['target_date'] for r in rebuilt[left]) < min(r['news_date'] for r in rebuilt[right])
    datasets = {split: PreparedDataset(ROOT, split) for split in rebuilt}
    feature = datasets['test']
    assert feature.text.shape == (32286, 384)
    assert np.isfinite(feature.text).all() and np.isfinite(feature.description).all()
    assert np.isfinite(feature.sentiment).all() and (feature.sentiment >= 0).all()
    assert np.allclose(feature.sentiment.sum(axis=1), 1, atol=1e-5)
    spec = importlib.util.spec_from_file_location('experiments', ROOT / 'scripts/05_run_experiments.py')
    experiments = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiments)
    fingerprint = experiments.run_fingerprint()
    combined = json.loads((ROOT / 'artifacts/runs/all_runs.json').read_text())
    assert combined['fingerprint'] == fingerprint
    expected = {('always_up', None)} | {(v, s) for v in VARIANTS for s in (17, 42, 73)}
    assert {(r['variant'], r.get('seed')) for r in combined['runs']} == expected
    assert len(combined['runs']) == len(expected)
    loaders = {split: DataLoader(dataset, batch_size=32, shuffle=False) for split, dataset in datasets.items()}
    targets = [r['target_up'] for r in rebuilt['test']]
    verified = []
    for record in combined['runs']:
        variant, seed = record['variant'], record.get('seed')
        stem = variant if seed is None else f'{variant}_{seed}'
        assert record == json.loads((ROOT / f'artifacts/runs/{stem}.json').read_text())
        assert record['fingerprint'] == fingerprint
        assert len(record['predictions']) == len(targets)
        for saved, row in zip(record['predictions'], rebuilt['test']):
            for key in ('news_date', 'forecast_date', 'target_date', 'target_up'):
                assert saved[key] == row[key]
        saved_scores = [p['probability_up'] for p in record['predictions']]
        assert classification_metrics(targets, saved_scores) == record['test_metrics']
        detail = {'variant': variant, 'seed': seed, 'test_metrics': record['test_metrics']}
        if seed is not None:
            model = make_model(variant).eval()
            checkpoint = ROOT / f'artifacts/runs/{stem}.pt'
            model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
            scores, weights = predict(model, loaders['test'], 'cpu')
            assert np.allclose(scores, saved_scores, atol=1e-6, rtol=0)
            valid_scores, _ = predict(model, loaders['valid'], 'cpu')
            valid_targets = [r['target_up'] for r in rebuilt['valid']]
            valid_metrics = classification_metrics(valid_targets, valid_scores)
            for key in ('accuracy', 'balanced_accuracy', 'log_loss'):
                assert abs(valid_metrics[key] - record['validation_metrics'][key]) < 1e-6
            replay_loss = evaluate_loss(model, loaders['valid'], 'cpu')
            assert abs(replay_loss - record['best_validation_loss']) < 1e-6
            best_loss = float('inf')
            best_epoch = 0
            for item in record['history']:
                if item['valid_loss'] < best_loss - 1e-5:
                    best_loss, best_epoch = item['valid_loss'], item['epoch']
            assert best_epoch == record['best_epoch']
            if weights and weights[0]:
                for row, saved, weight in zip(rebuilt['test'], record['predictions'], weights):
                    n = len(row['headlines'])
                    assert abs(sum(weight[:n]) - 1) < 1e-5 and all(w == 0 for w in weight[n:])
                    top = max(range(n), key=lambda i: weight[i])
                    assert row['headlines'][top]['id'] == saved['top_headline_id']
                    assert abs(weight[top] - saved['top_attention_weight']) < 1e-6
            detail.update(checkpoint_sha256=sha256_file(checkpoint),
                          max_probability_error=float(np.max(np.abs(np.array(scores) - saved_scores))),
                          parameter_count=sum(p.numel() for p in model.parameters()),
                          best_epoch=best_epoch,
                          predicted_up=sum(p >= .5 for p in scores))
        verified.append(detail)
        print(f'Verified {stem}', flush=True)
    prior = sum(r['target_up'] for r in rebuilt['train']) / len(rebuilt['train'])
    baseline = {'probability_up': prior, 'fitted_on': 'training targets only',
                'test_metrics': classification_metrics(targets, [prior] * len(targets))}
    # A small training-batch optimization check separates a functioning optimizer from
    # the empirical absence of a useful signal in the historical experiment.
    torch.manual_seed(123)
    model = make_model('full')
    batch = next(iter(DataLoader(datasets['train'], batch_size=8, shuffle=False)))
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    model.eval()
    initial = torch.nn.functional.binary_cross_entropy_with_logits(model(batch)[0], batch['target']).item()
    for _ in range(150):
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(batch)[0], batch['target'])
        loss.backward()
        optimizer.step()
    final = torch.nn.functional.binary_cross_entropy_with_logits(model(batch)[0], batch['target']).item()
    assert final < .1 and final < initial * .25
    report = {'audited_at_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'PASS', 'source_returns_checked': source_count,
              'examples_rebuilt': {s: len(r) for s, r in rebuilt.items()},
              'feature_cache_verified': True, 'original_fingerprint': fingerprint,
              'checkpoint_runs_replayed': 18, 'verified_runs': verified,
              'training_prior_reference': baseline,
              'tiny_batch_overfit': {'rows': 8, 'steps': 150, 'initial_loss': initial, 'final_loss': final},
              'environment': {'python': sys.version.split()[0], 'torch': torch.__version__, 'numpy': np.__version__},
              'scope': 'Retrospective architecture prototype. No retraining or test-driven model changes.'}
    output = ROOT / 'reports/prototype_audit.json'
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    print(json.dumps({k: v for k, v in report.items() if k != 'verified_runs'}, indent=2))


if __name__ == '__main__':
    main()

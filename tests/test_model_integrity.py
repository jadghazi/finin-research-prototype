"""Check that attention respects padding and the model actually learns through its core blocks."""

from __future__ import annotations

import unittest

import torch

from src.evaluation import classification_metrics
from src.models.finin import ReducedFININ


def synthetic_batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(2)
    return {
        "market_features": torch.randn(2, 6),
        "market_description": torch.randn(384),
        "headline_vectors": torch.randn(2, 4, 384),
        "sentiment": torch.softmax(torch.randn(2, 4, 3), dim=-1),
        "mask": torch.tensor([[True, True, False, False], [True, True, True, False]]),
    }


class ModelIntegrityTest(unittest.TestCase):
    def test_padding_does_not_change_prediction_or_weights(self) -> None:
        model = ReducedFININ(dropout=0.0).eval()
        original = synthetic_batch()
        changed = {key: value.clone() for key, value in original.items()}
        changed["headline_vectors"][~original["mask"]] = 500
        changed["sentiment"][~original["mask"]] = -500
        with torch.inference_mode():
            logits_a, weights_a = model(original)
            logits_b, weights_b = model(changed)
        torch.testing.assert_close(logits_a, logits_b, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(weights_a, weights_b, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(weights_a.sum(dim=1), torch.ones(2))
        self.assertTrue(torch.all(weights_a[~original["mask"]] == 0))

    def test_training_gradient_reaches_both_attention_components(self) -> None:
        model = ReducedFININ(dropout=0.0)
        batch = synthetic_batch()
        logits, _ = model(batch)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([1., 0.]))
        loss.backward()
        for parameter in (
            model.encoders.text.weight,
            model.attention.self_attention.in_proj_weight,
            model.attention.query.weight,
            model.attention.key.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_always_up_balanced_accuracy_exposes_single_class_guessing(self) -> None:
        metrics = classification_metrics([0, 1, 0, 1], [1, 1, 1, 1])
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["balanced_accuracy"], 0.5)
        self.assertEqual(metrics["confusion"]["false_up"], 2)


if __name__ == "__main__":
    unittest.main()

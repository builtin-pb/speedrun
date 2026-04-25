from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from simple_model import GPT, GPTConfig
from train_logging import collect_norm_metrics, collect_stability_metrics


class _TinyLoggingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(2, 2)
        self.proj = nn.Linear(2, 2, bias=False)


class _DummyStabilityModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(logit_softcap=15.0)

    def compute_raw_logits(self, inputs: torch.Tensor, observer=None) -> torch.Tensor:
        activation = torch.tensor(
            [[[3.0, 4.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
            dtype=torch.float32,
        )
        update = torch.tensor(
            [[[6.0, 8.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
            dtype=torch.float32,
        )
        residual = torch.ones((1, 2, 4), dtype=torch.float32)
        matrix_activation = torch.full((1, 2, 4), 2.0, dtype=torch.float32)
        if observer is not None:
            observer("matrix_embed/act_abs", matrix_activation)
            observer("layer_embed/activation_rms", activation)
            observer("layer_attn/block_00_update_rms", update)
            observer("layer_final/residual_rms", residual)
        return torch.tensor(
            [[[9.0, -9.0, 8.0, 7.0, 6.0], [5.0, -8.0, -7.0, -6.0, -5.0]]],
            dtype=torch.float32,
        )


class _DummyGateStabilityModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(logit_softcap=15.0)

    def compute_raw_logits(self, inputs: torch.Tensor, observer=None) -> torch.Tensor:
        if observer is not None:
            observer("gate_coeff_attn/block_00", torch.tensor([[[0.5], [1.0]]], dtype=torch.float32))
            observer("gate_coeff_mlp/block_00", torch.tensor([[[1.5], [2.0]]], dtype=torch.float32))
        return torch.zeros((1, 2, 5), dtype=torch.float32)


class TrainLoggingTests(unittest.TestCase):
    def test_collect_norm_metrics_reports_rms_for_global_and_matrix_metrics(self) -> None:
        model = _TinyLoggingModel()
        with torch.no_grad():
            model.embed.weight.copy_(torch.tensor([[3.0, 4.0], [0.0, 0.0]]))
            model.proj.weight.copy_(torch.tensor([[1.0, 2.0], [2.0, 1.0]]))

        model.embed.weight.grad = torch.tensor([[0.0, 6.0], [8.0, 0.0]])
        model.proj.weight.grad = torch.tensor([[1.0, 2.0], [2.0, 1.0]])

        main_metrics, matrix_metrics = collect_norm_metrics(model, include_matrix=True)

        self.assertAlmostEqual(matrix_metrics["matrix_embed/param_rms"], math.sqrt(25.0 / 4.0))
        self.assertAlmostEqual(matrix_metrics["matrix_lm_head/param_rms"], math.sqrt(10.0 / 4.0))
        self.assertAlmostEqual(matrix_metrics["matrix_embed/grad_rms"], math.sqrt(100.0 / 4.0))
        self.assertAlmostEqual(matrix_metrics["matrix_lm_head/grad_rms"], math.sqrt(10.0 / 4.0))
        self.assertAlmostEqual(main_metrics["main/global_param_rms"], math.sqrt(35.0 / 8.0), places=6)
        self.assertAlmostEqual(main_metrics["main/global_grad_rms"], math.sqrt(110.0 / 8.0), places=6)
        self.assertNotIn("main/global_param_l2", main_metrics)
        self.assertNotIn("main/global_grad_l2", main_metrics)
        self.assertNotIn("matrix_embed/param_l2", matrix_metrics)
        self.assertNotIn("matrix_lm_head/grad_l2", matrix_metrics)

    def test_collect_norm_metrics_groups_residual_gate_parameters(self) -> None:
        model = GPT(
            GPTConfig(
                vocab_size=16,
                num_layers=1,
                model_dim=4,
                head_dim=4,
                mlp_expansion=1,
                learned_residual_gates=True,
                gate_feature_dim=2,
            )
        )
        for param in model.parameters():
            param.grad = torch.ones_like(param)

        _, matrix_metrics = collect_norm_metrics(model, include_matrix=True)

        self.assertIn("gate_trunk/block_00_param_rms", matrix_metrics)
        self.assertIn("gate_trunk/block_00_grad_rms", matrix_metrics)
        self.assertIn("gate_attn_head/block_00_weight_param_rms", matrix_metrics)
        self.assertIn("gate_attn_head/block_00_bias_param_rms", matrix_metrics)
        self.assertIn("gate_mlp_head/block_00_weight_param_rms", matrix_metrics)
        self.assertIn("gate_mlp_head/block_00_bias_param_rms", matrix_metrics)
        self.assertFalse(
            any("residual_gate" in metric_name and metric_name.startswith("matrix_other/") for metric_name in matrix_metrics)
        )

    def test_collect_stability_metrics_reports_reduced_metrics_without_logits_max_abs(self) -> None:
        model = _DummyStabilityModel()
        inputs = torch.zeros((1, 2), dtype=torch.long)

        metrics = collect_stability_metrics(
            model,
            inputs,
            micro_batch_size=1,
            max_sequences=1,
            rank=0,
        )

        self.assertAlmostEqual(metrics["layer_embed/activation_rms"], math.sqrt(25.0 / 8.0))
        self.assertAlmostEqual(metrics["layer_attn/block_00_update_rms"], math.sqrt(100.0 / 8.0))
        self.assertAlmostEqual(metrics["layer_final/residual_rms"], 1.0, places=6)
        self.assertEqual(metrics["logits/top1"], 9.0)
        self.assertEqual(metrics["logits/top5_mean"], 7.0)
        self.assertEqual(metrics["logits/bottom1"], -9.0)
        self.assertEqual(metrics["logits/bottom5_mean"], -7.0)
        self.assertEqual(metrics["matrix_embed/act_abs_p50"], 2.0)
        self.assertEqual(metrics["matrix_embed/act_abs_p90"], 2.0)
        self.assertEqual(metrics["matrix_embed/act_abs_p99"], 2.0)
        self.assertNotIn("layer_embed/activation_l2", metrics)
        self.assertNotIn("layer_attn/block_00_update_l2", metrics)
        self.assertNotIn("layer_final/residual_l2", metrics)
        self.assertNotIn("logits/max_abs", metrics)

    def test_simple_model_emits_rms_names_for_requested_observer_metrics(self) -> None:
        model = GPT(GPTConfig(vocab_size=16, num_layers=1, model_dim=4, head_dim=4, mlp_expansion=1))
        seen: list[str] = []

        model.compute_raw_logits(
            torch.zeros((1, 2), dtype=torch.long),
            observer=lambda name, tensor: seen.append(name),
        )

        self.assertIn("layer_embed/activation_rms", seen)
        self.assertIn("layer_attn/block_00_update_rms", seen)
        self.assertIn("layer_mlp/block_00_update_rms", seen)
        self.assertIn("layer_final/residual_rms", seen)
        self.assertNotIn("layer_embed/activation_l2", seen)
        self.assertNotIn("layer_attn/block_00_update_l2", seen)
        self.assertNotIn("layer_mlp/block_00_update_l2", seen)
        self.assertNotIn("layer_final/residual_l2", seen)
        self.assertIn("layer_attn/block_00_input_rms", seen)
        self.assertIn("layer_attn/block_00_output_rms", seen)
        self.assertIn("layer_mlp/block_00_input_rms", seen)
        self.assertIn("layer_mlp/block_00_output_rms", seen)
        self.assertIn("matrix_embed/act_abs", seen)
        self.assertIn("matrix_attn_q/block_00_act_abs", seen)
        self.assertIn("matrix_attn_k/block_00_act_abs", seen)
        self.assertIn("matrix_attn_v/block_00_act_abs", seen)
        self.assertIn("matrix_attn_proj/block_00_act_abs", seen)
        self.assertIn("matrix_mlp_fc/block_00_act_abs", seen)
        self.assertIn("matrix_mlp_proj/block_00_act_abs", seen)
        self.assertNotIn("layer_attn/block_00_input_l2", seen)
        self.assertNotIn("layer_attn/block_00_output_l2", seen)
        self.assertNotIn("layer_mlp/block_00_input_l2", seen)
        self.assertNotIn("layer_mlp/block_00_output_l2", seen)

    def test_simple_model_emits_gate_coefficients_when_enabled(self) -> None:
        model = GPT(
            GPTConfig(
                vocab_size=16,
                num_layers=1,
                model_dim=4,
                head_dim=4,
                mlp_expansion=1,
                learned_residual_gates=True,
                gate_feature_dim=2,
            )
        )
        seen: list[str] = []

        model.compute_raw_logits(
            torch.zeros((1, 2), dtype=torch.long),
            observer=lambda name, tensor: seen.append(name),
        )

        self.assertIn("gate_coeff_attn/block_00", seen)
        self.assertIn("gate_coeff_mlp/block_00", seen)

    def test_collect_stability_metrics_reports_gate_coefficient_summaries(self) -> None:
        model = _DummyGateStabilityModel()
        inputs = torch.zeros((1, 2), dtype=torch.long)

        metrics = collect_stability_metrics(
            model,
            inputs,
            micro_batch_size=1,
            max_sequences=1,
            rank=0,
        )

        self.assertEqual(metrics["gate_coeff_attn/block_00_mean"], 0.75)
        self.assertAlmostEqual(metrics["gate_coeff_attn/block_00_std"], 0.25)
        self.assertEqual(metrics["gate_coeff_attn/block_00_min"], 0.5)
        self.assertEqual(metrics["gate_coeff_attn/block_00_max"], 1.0)
        self.assertEqual(metrics["gate_coeff_attn/block_00_delta_abs_mean"], 0.25)
        self.assertEqual(metrics["gate_coeff_mlp/block_00_mean"], 1.75)
        self.assertAlmostEqual(metrics["gate_coeff_mlp/block_00_std"], 0.25)
        self.assertEqual(metrics["gate_coeff_mlp/block_00_delta_abs_mean"], 0.75)
        self.assertEqual(metrics["gate_coeff_summary/attn_mean"], 0.75)
        self.assertEqual(metrics["gate_coeff_summary/mlp_mean"], 1.75)

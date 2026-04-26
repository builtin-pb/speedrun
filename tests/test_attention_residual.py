from __future__ import annotations

import unittest
import math

import torch
from torch import nn

from simple_model import FullAttentionResidual, GPT, GPTConfig, norm
from simple_optim import build_optimizers
from train_gpt_simple import build_parser


class AttentionResidualTests(unittest.TestCase):
    def test_full_attention_residual_is_opt_in_and_can_be_disabled(self) -> None:
        parser = build_parser()

        self.assertFalse(parser.parse_args([]).attention_residual)
        self.assertTrue(parser.parse_args(["--attention-residual"]).attention_residual)
        self.assertFalse(parser.parse_args(["--no-attention-residual"]).attention_residual)

    def test_full_attention_residual_adds_zero_initialized_depth_queries(self) -> None:
        model = GPT(
            GPTConfig(
                vocab_size=32,
                num_layers=2,
                model_dim=8,
                head_dim=4,
                mlp_expansion=1,
                attention_residual=True,
            )
        )

        self.assertTrue(model.config.attention_residual)
        self.assertEqual(tuple(model.final_res.query.weight.shape), (8,))
        self.assertIsNone(model.blocks[0].attn_res)
        self.assertEqual(tuple(model.blocks[0].mlp_res.query.weight.shape), (8,))
        self.assertEqual(tuple(model.blocks[1].attn_res.query.weight.shape), (8,))
        self.assertEqual(float(model.final_res.query.weight.abs().sum().item()), 0.0)
        self.assertEqual(float(model.blocks[0].mlp_res.query.weight.abs().sum().item()), 0.0)
        self.assertEqual(float(model.blocks[1].attn_res.query.weight.abs().sum().item()), 0.0)

    def test_zero_query_attention_residual_uniformly_averages_full_history(self) -> None:
        attn_res = FullAttentionResidual(dim=2)
        history = [
            torch.tensor([[[1.0, 3.0]]]),
            torch.tensor([[[5.0, 7.0]]]),
            torch.tensor([[[9.0, 11.0]]]),
        ]

        out = attn_res(history)

        self.assertTrue(torch.allclose(out, torch.tensor([[[5.0, 7.0]]])))

    def test_nonzero_query_attention_residual_matches_depth_softmax(self) -> None:
        attn_res = FullAttentionResidual(dim=2)
        with torch.no_grad():
            attn_res.query.weight.copy_(torch.tensor([1.0, 0.0]))
        history = [
            torch.tensor([[[1.0, 0.0]]]),
            torch.tensor([[[0.0, 1.0]]]),
        ]
        logits = torch.stack([(norm(state).float() * attn_res.query.weight.float()).sum(dim=-1) for state in history])
        weights = logits.softmax(dim=0)
        expected = weights[0].unsqueeze(-1) * history[0] + weights[1].unsqueeze(-1) * history[1]

        out = attn_res(history)

        self.assertTrue(torch.allclose(out, expected))

    def test_zero_query_final_attention_residual_reports_source_weight_buckets(self) -> None:
        attn_res = FullAttentionResidual(dim=2)
        history = [
            torch.tensor([[[1.0, 0.0]]]),
            torch.tensor([[[2.0, 0.0]]]),
            torch.tensor([[[3.0, 0.0]]]),
            torch.tensor([[[4.0, 0.0]]]),
            torch.tensor([[[5.0, 0.0]]]),
        ]
        observed: dict[str, torch.Tensor] = {}

        attn_res(
            history,
            observer=lambda name, tensor: observed.setdefault(name, tensor.detach().clone()),
            metric_prefix="attnres_final/output",
        )

        self.assertTrue(torch.allclose(observed["attnres_final/output_embedding_weight_rms"], torch.full((1, 1), 0.2)))
        self.assertTrue(torch.allclose(observed["attnres_final/output_attn_source_weight_rms"], torch.full((1, 1), 0.4)))
        self.assertTrue(torch.allclose(observed["attnres_final/output_mlp_source_weight_rms"], torch.full((1, 1), 0.4)))
        self.assertTrue(torch.allclose(observed["attnres_final/output_current_weight_rms"], torch.full((1, 1), 0.2)))
        self.assertTrue(torch.allclose(observed["attnres_final/output_weight_entropy_rms"], torch.full((1, 1), math.log(5))))

    def test_attention_residual_queries_use_adam_not_muon(self) -> None:
        model = GPT(
            GPTConfig(
                vocab_size=32,
                num_layers=2,
                model_dim=8,
                head_dim=4,
                mlp_expansion=1,
                attention_residual=True,
            )
        )
        optimizers = build_optimizers(model, fused_adamw=False)
        query_params = {
            id(param)
            for name, param in model.named_parameters()
            if name.endswith(("attn_res.query.weight", "mlp_res.query.weight"))
            or name == "final_res.query.weight"
        }
        adam_params = {id(param) for group in optimizers[0].param_groups for param in group["params"]}
        muon_params = {id(param) for group in optimizers[1].param_groups for param in group["params"]}

        self.assertTrue(query_params)
        self.assertTrue(query_params <= adam_params)
        self.assertFalse(query_params & muon_params)

    def test_block_attention_residual_uses_expected_history_lengths(self) -> None:
        torch.manual_seed(0)

        class ConstantUpdate(nn.Module):
            def __init__(self, value: float) -> None:
                super().__init__()
                self.value = value

            def forward(self, x, *args, **kwargs):
                return torch.full_like(x, self.value)

        class RecordingResidual(nn.Module):
            def __init__(
                self,
                label: str,
                calls: list[tuple[str, int]],
                snapshots: dict[str, list[torch.Tensor]],
            ) -> None:
                super().__init__()
                self.label = label
                self.calls = calls
                self.snapshots = snapshots

            def forward(self, history, observer=None, metric_prefix=None):
                self.calls.append((self.label, len(history)))
                self.snapshots[self.label] = [state.detach().clone() for state in history]
                return history[-1]

        model = GPT(
            GPTConfig(
                vocab_size=32,
                num_layers=2,
                model_dim=8,
                head_dim=4,
                mlp_expansion=1,
                attention_residual=True,
            )
        )
        calls: list[tuple[str, int]] = []
        snapshots: dict[str, list[torch.Tensor]] = {}
        model.blocks[0].attn = ConstantUpdate(1.0)
        model.blocks[0].mlp = ConstantUpdate(2.0)
        model.blocks[1].attn = ConstantUpdate(3.0)
        model.blocks[1].mlp = ConstantUpdate(4.0)
        model.blocks[0].mlp_res = RecordingResidual("block0_mlp", calls, snapshots)
        model.blocks[1].attn_res = RecordingResidual("block1_attn", calls, snapshots)
        model.blocks[1].mlp_res = RecordingResidual("block1_mlp", calls, snapshots)
        model.final_res = RecordingResidual("final", calls, snapshots)

        inputs = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        embedding_source = norm(model.embed(inputs))

        model.compute_raw_logits(inputs)

        self.assertEqual(
            calls,
            [("block0_mlp", 2), ("block1_attn", 3), ("block1_mlp", 4), ("final", 5)],
        )
        attn0_source = torch.full_like(embedding_source, 1.0)
        mlp0_source = torch.full_like(embedding_source, 2.0)
        attn1_source = torch.full_like(embedding_source, 3.0)
        expected_sources = {
            "block0_mlp": [embedding_source, attn0_source],
            "block1_attn": [embedding_source, attn0_source, mlp0_source],
            "block1_mlp": [embedding_source, attn0_source, mlp0_source, attn1_source],
            "final": [embedding_source, attn0_source, mlp0_source, attn1_source, mlp0_source.new_full(mlp0_source.shape, 4.0)],
        }
        for label, expected in expected_sources.items():
            self.assertEqual(len(snapshots[label]), len(expected))
            for actual, expected_source in zip(snapshots[label], expected):
                self.assertTrue(torch.allclose(actual, expected_source))

    def test_final_projection_uses_final_attention_residual_output(self) -> None:
        class ConstantUpdate(nn.Module):
            def __init__(self, value: float) -> None:
                super().__init__()
                self.value = value

            def forward(self, x, *args, **kwargs):
                return torch.full_like(x, self.value)

        class FinalResidual(nn.Module):
            def __init__(self, output: torch.Tensor) -> None:
                super().__init__()
                self.output = output

            def forward(self, history, observer=None, metric_prefix=None):
                return self.output.to(device=history[-1].device, dtype=history[-1].dtype)

        model = GPT(
            GPTConfig(
                vocab_size=8,
                num_layers=1,
                model_dim=4,
                head_dim=4,
                mlp_expansion=1,
                attention_residual=True,
            )
        )
        model.blocks[0].attn = ConstantUpdate(1.0)
        model.blocks[0].mlp = ConstantUpdate(2.0)
        final_output = torch.tensor([[[-1.0, 0.0, 0.0, 0.0]]], dtype=torch.bfloat16)
        model.final_res = FinalResidual(final_output)
        with torch.no_grad():
            model.proj.weight.zero_()
            model.proj.weight[3, 0] = 1.0

        logits = model.compute_raw_logits(torch.tensor([[0]], dtype=torch.long))

        detached_logits = logits.detach()
        self.assertLess(float(detached_logits[0, 0, 3]), -0.9)
        self.assertTrue(torch.allclose(detached_logits[0, 0, :3], torch.zeros(3)))

    def test_block_attention_residual_rejects_wrong_history_length(self) -> None:
        model = GPT(
            GPTConfig(
                vocab_size=32,
                num_layers=2,
                model_dim=8,
                head_dim=4,
                mlp_expansion=1,
                attention_residual=True,
            )
        )
        x = torch.randn(1, 2, 8)

        with self.assertRaisesRegex(ValueError, "history for block 1 must have length 3"):
            model.blocks[1](x, history=[x])

    def test_full_attention_residual_changes_the_residual_path_when_enabled(self) -> None:
        torch.manual_seed(0)
        baseline = GPT(
            GPTConfig(
                vocab_size=32,
                num_layers=1,
                model_dim=8,
                head_dim=4,
                mlp_expansion=1,
                attention_residual=False,
            )
        )
        torch.manual_seed(0)
        attnres = GPT(
            GPTConfig(
                vocab_size=32,
                num_layers=1,
                model_dim=8,
                head_dim=4,
                mlp_expansion=1,
                attention_residual=True,
            )
        )
        shared_state = {
            name: value
            for name, value in baseline.state_dict().items()
            if name in attnres.state_dict()
        }
        attnres.load_state_dict(shared_state, strict=False)

        inputs = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

        self.assertFalse(torch.allclose(baseline.compute_raw_logits(inputs), attnres.compute_raw_logits(inputs)))


if __name__ == "__main__":
    unittest.main()

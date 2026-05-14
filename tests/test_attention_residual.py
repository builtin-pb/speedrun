from __future__ import annotations

import unittest
import math
import contextlib
import io

import torch
from torch import nn

from simple_model import FullAttentionResidual, GPT, GPTConfig, norm
from simple_optim import build_optimizers
from train_gpt_simple import build_parser, validate_args


class AttentionResidualTests(unittest.TestCase):
    def test_full_attention_residual_is_opt_in_and_can_be_disabled(self) -> None:
        parser = build_parser()

        self.assertFalse(parser.parse_args([]).attention_residual)
        self.assertTrue(parser.parse_args(["--attention-residual"]).attention_residual)
        self.assertFalse(parser.parse_args(["--no-attention-residual"]).attention_residual)

    def test_attention_residual_flags_parse_and_default_lr_scalar(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
                "--model-dim",
                "768",
                "--attnres-logit-scale",
                "none",
                "--attnres-normalize-values",
                "--adam-attnres-lr",
                "2.0",
                "--adam-attnres-lr-scale",
                "dim",
            ]
        )
        validate_args(args)

        self.assertEqual(args.attnres_logit_scale, "none")
        self.assertTrue(args.attnres_normalize_values)
        self.assertEqual(args.adam_attnres_lr, 2.0)
        self.assertEqual(args.adam_attnres_lr_scale, "dim")

        default_lr_args = parser.parse_args(["--model-dim", "8", "--head-dim", "4"])
        validate_args(default_lr_args)

        self.assertEqual(default_lr_args.attnres_logit_scale, "none")
        self.assertFalse(default_lr_args.attnres_normalize_values)
        self.assertEqual(default_lr_args.adam_attnres_lr, 1.0)
        self.assertEqual(default_lr_args.adam_attnres_lr_scale, "dim")

    def test_attention_residual_logit_and_lr_scales_must_match(self) -> None:
        parser = build_parser()

        matching = [
            ("none", "dim"),
            ("sqrt_dim", "sqrt_dim"),
            ("dim", "none"),
        ]
        for logit_scale, lr_scale in matching:
            args = parser.parse_args(
                [
                    "--model-dim",
                    "8",
                    "--head-dim",
                    "4",
                    "--attnres-logit-scale",
                    logit_scale,
                    "--adam-attnres-lr-scale",
                    lr_scale,
                ]
            )
            validate_args(args)

        mismatched_args = parser.parse_args(
            [
                "--model-dim",
                "8",
                "--head-dim",
                "4",
                "--attnres-logit-scale",
                "sqrt_dim",
                "--adam-attnres-lr-scale",
                "dim",
            ]
        )
        with self.assertRaisesRegex(ValueError, "must match"):
            validate_args(mismatched_args)

    def test_removed_attention_residual_modes_are_not_cli_flags(self) -> None:
        parser = build_parser()

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--attnres-history-mode", "output"])
            with self.assertRaises(SystemExit):
                parser.parse_args(["--attnres-final-mode", "mixer"])
            with self.assertRaises(SystemExit):
                parser.parse_args(["--attnres-logit-scale-mode", "none"])
            with self.assertRaises(SystemExit):
                parser.parse_args(["--attnres-logit-scale", "auto"])
            with self.assertRaises(SystemExit):
                parser.parse_args(["--attnres-logit-scale", "sqrt-dim"])
            with self.assertRaises(SystemExit):
                parser.parse_args(["--attnres-logit-scale", "inverse_sqrt_dim"])
            with self.assertRaises(SystemExit):
                parser.parse_args(["--adam-attnres-lr-mode", "1/sqrt(d)"])
            with self.assertRaises(SystemExit):
                parser.parse_args(["--adam-attnres-lr-scale", "1/sqrt(d)"])

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

    def test_none_logit_scale_leaves_attention_residual_logits_unscaled(self) -> None:
        attn_res = FullAttentionResidual(dim=2, logit_scale_mode="none")
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

    def test_attention_residual_can_normalize_values(self) -> None:
        attn_res = FullAttentionResidual(dim=2, normalize_values=True)
        history = [
            torch.tensor([[[3.0, 0.0]]]),
            torch.tensor([[[0.0, 4.0]]]),
        ]
        expected = 0.5 * norm(history[0]) + 0.5 * norm(history[1])

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
        self.assertEqual(optimizers[0].param_groups[2]["lr"], 1 / model.config.model_dim)

        sqrt_scaled_model = GPT(
            GPTConfig(
                vocab_size=32,
                num_layers=2,
                model_dim=8,
                head_dim=4,
                mlp_expansion=1,
                attention_residual=True,
                attnres_logit_scale="sqrt_dim",
            )
        )
        scaled_optimizers = build_optimizers(
            sqrt_scaled_model,
            adam_attnres_lr=2.0,
            adam_attnres_lr_scale="sqrt_dim",
            fused_adamw=False,
        )
        dim_scaled_model = GPT(
            GPTConfig(
                vocab_size=32,
                num_layers=2,
                model_dim=8,
                head_dim=4,
                mlp_expansion=1,
                attention_residual=True,
                attnres_logit_scale="dim",
            )
        )
        unscaled_optimizers = build_optimizers(
            dim_scaled_model,
            adam_attnres_lr=2.0,
            adam_attnres_lr_scale="none",
            fused_adamw=False,
        )
        self.assertEqual(scaled_optimizers[0].param_groups[2]["lr"], 2.0 / math.sqrt(model.config.model_dim))
        self.assertEqual(unscaled_optimizers[0].param_groups[2]["lr"], 2.0)

        with self.assertRaisesRegex(ValueError, "must match"):
            build_optimizers(
                model,
                adam_attnres_lr_scale="sqrt_dim",
                fused_adamw=False,
            )

    def test_lm_head_lr_is_model_dim_scaled(self) -> None:
        model = GPT(
            GPTConfig(
                vocab_size=32,
                num_layers=2,
                model_dim=384,
                head_dim=64,
                mlp_expansion=4,
            )
        )

        optimizers = build_optimizers(model, fused_adamw=False)

        self.assertEqual(optimizers[0].param_groups[0]["lr"], (768 / 320) / model.config.model_dim)

    def test_mlp_projection_uses_smaller_muon_lr(self) -> None:
        model = GPT(
            GPTConfig(
                vocab_size=32,
                num_layers=2,
                model_dim=16,
                head_dim=4,
                mlp_expansion=4,
            )
        )

        optimizers = build_optimizers(model, muon_lr=0.02, fused_adamw=False)
        mlp_proj_params = {id(block.mlp.proj.weight) for block in model.blocks}
        groups_by_lr = {
            group["lr"]: {id(param) for param in group["params"]}
            for group in optimizers[1].param_groups
        }

        self.assertEqual(groups_by_lr[0.01], mlp_proj_params)
        self.assertTrue(groups_by_lr[0.02].isdisjoint(mlp_proj_params))

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

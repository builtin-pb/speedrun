from __future__ import annotations

import unittest
import sys
import types
from contextlib import redirect_stderr
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

from train_gpt_simple import build_gpt_config, build_parser, validate_args


class TrainArgsTests(unittest.TestCase):
    def test_layer_norm_position_defaults_to_pre_norm(self) -> None:
        args = build_parser().parse_args([])

        self.assertEqual(args.layer_norm_position, "pre")

    def test_layer_norm_position_accepts_post_norm(self) -> None:
        args = build_parser().parse_args(["--layer-norm-position", "post"])

        self.assertEqual(args.layer_norm_position, "post")

    def test_layer_norm_position_rejects_unknown_values(self) -> None:
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["--layer-norm-position", "sandwich"])

    def test_residual_depth_scale_defaults_to_init(self) -> None:
        args = build_parser().parse_args([])

        self.assertEqual(args.residual_depth_scale, "init")

    def test_residual_depth_scale_accepts_none_and_forward(self) -> None:
        for mode in ("none", "forward"):
            with self.subTest(mode=mode):
                args = build_parser().parse_args(["--residual-depth-scale", mode])

                self.assertEqual(args.residual_depth_scale, mode)

    def test_residual_depth_scale_rejects_unknown_values(self) -> None:
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["--residual-depth-scale", "sometimes"])

    def test_build_gpt_config_passes_norm_and_residual_scale_knobs(self) -> None:
        args = build_parser().parse_args(
            [
                "--layer-norm-position",
                "post",
                "--residual-depth-scale",
                "none",
                "--embed-init-scale",
                "0.25",
                "--lm-head-init-scale",
                "1.5",
                "--attn-q-init-scale",
                "2.0",
                "--attn-k-init-scale",
                "0.5",
                "--attn-v-init-scale",
                "1.25",
                "--attn-proj-init-scale",
                "0.75",
                "--mlp-fc-init-scale",
                "1.75",
                "--mlp-proj-init-scale",
                "0.6",
            ]
        )
        fake_simple_model = types.ModuleType("simple_model")
        fake_simple_model.GPTConfig = lambda **kwargs: SimpleNamespace(**kwargs)

        with patch.dict(sys.modules, {"simple_model": fake_simple_model}):
            config = build_gpt_config(args)

        self.assertEqual(config.layer_norm_position, "post")
        self.assertEqual(config.residual_depth_scale, "none")
        self.assertEqual(config.embed_init_scale, 0.25)
        self.assertEqual(config.lm_head_init_scale, 1.5)
        self.assertEqual(config.attn_q_init_scale, 2.0)
        self.assertEqual(config.attn_k_init_scale, 0.5)
        self.assertEqual(config.attn_v_init_scale, 1.25)
        self.assertEqual(config.attn_proj_init_scale, 0.75)
        self.assertEqual(config.mlp_fc_init_scale, 1.75)
        self.assertEqual(config.mlp_proj_init_scale, 0.6)

    def test_init_scale_knobs_default_to_one(self) -> None:
        args = build_parser().parse_args([])

        self.assertEqual(args.embed_init_scale, 1.0)
        self.assertEqual(args.lm_head_init_scale, 1.0)
        self.assertEqual(args.attn_q_init_scale, 1.0)
        self.assertEqual(args.attn_k_init_scale, 1.0)
        self.assertEqual(args.attn_v_init_scale, 1.0)
        self.assertEqual(args.attn_proj_init_scale, 1.0)
        self.assertEqual(args.mlp_fc_init_scale, 1.0)
        self.assertEqual(args.mlp_proj_init_scale, 1.0)

    def test_init_scale_knobs_reject_negative_values(self) -> None:
        args = build_parser().parse_args(["--mlp-proj-init-scale", "-0.1"])

        with self.assertRaisesRegex(ValueError, "--mlp-proj-init-scale"):
            validate_args(args)

    def test_muon_schedule_knobs_accept_override_and_floor(self) -> None:
        args = build_parser().parse_args(["--muon-cooldown-frac", "0.3", "--muon-lr-floor-scale", "0.2"])

        self.assertEqual(args.muon_cooldown_frac, 0.3)
        self.assertEqual(args.muon_lr_floor_scale, 0.2)
        validate_args(args)

    def test_muon_residual_knobs_accept_valid_values(self) -> None:
        args = build_parser().parse_args(["--muon-residual-lr-scale", "0.7", "--muon-residual-momentum", "0.8"])

        self.assertEqual(args.muon_residual_lr_scale, 0.7)
        self.assertEqual(args.muon_residual_momentum, 0.8)
        validate_args(args)

    def test_muon_schedule_knobs_reject_invalid_floor(self) -> None:
        args = build_parser().parse_args(["--muon-lr-floor-scale", "1.1"])

        with self.assertRaisesRegex(ValueError, "--muon-lr-floor-scale"):
            validate_args(args)

    def test_muon_residual_knobs_reject_invalid_values(self) -> None:
        args = build_parser().parse_args(["--muon-residual-lr-scale", "0.0"])

        with self.assertRaisesRegex(ValueError, "--muon-residual-lr-scale"):
            validate_args(args)

        args = build_parser().parse_args(["--muon-residual-momentum", "1.0"])

        with self.assertRaisesRegex(ValueError, "--muon-residual-momentum"):
            validate_args(args)

    def test_muon_schedule_knobs_reject_overlapping_muon_schedule(self) -> None:
        args = build_parser().parse_args(["--warmup-frac", "0.8", "--cooldown-frac", "0.1", "--muon-cooldown-frac", "0.3"])

        with self.assertRaisesRegex(ValueError, "must not exceed 1"):
            validate_args(args)

    def test_muon_schedule_knobs_accept_absolute_decay_override(self) -> None:
        args = build_parser().parse_args(["--muon-decay-start-step", "100", "--muon-decay-steps", "50"])

        self.assertEqual(args.muon_decay_start_step, 100)
        self.assertEqual(args.muon_decay_steps, 50)
        validate_args(args)

    def test_muon_schedule_knobs_reject_partial_absolute_decay_override(self) -> None:
        args = build_parser().parse_args(["--muon-decay-start-step", "100"])

        with self.assertRaisesRegex(ValueError, "provided together"):
            validate_args(args)

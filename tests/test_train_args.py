from __future__ import annotations

import unittest

from train_gpt_simple import build_parser
from train_gpt_simple import validate_args


class TrainArgsTests(unittest.TestCase):
    def test_learned_residual_gates_defaults_off_and_can_be_enabled(self) -> None:
        default_args = build_parser().parse_args([])
        gated_args = build_parser().parse_args(["--learned-residual-gates"])

        self.assertFalse(default_args.learned_residual_gates)
        self.assertTrue(gated_args.learned_residual_gates)

    def test_gate_tuning_args_have_defaults_and_can_be_overridden(self) -> None:
        default_args = build_parser().parse_args([])
        tuned_args = build_parser().parse_args([
            "--gate-feature-dim", "24",
            "--gate-delta-scale", "0.75",
            "--gate-warmup-frac", "0.1",
            "--gate-cooldown-frac", "0.2",
            "--gate-freeze-steps", "50",
            "--gate-lr", "0.006",
            "--gate-beta2", "0.997",
            "--gate-weight-decay", "0.01",
            "--gate-trunk-lr", "0.002",
            "--gate-trunk-beta2", "0.999",
            "--gate-trunk-weight-decay", "0.03",
            "--gate-bias-lr", "0.002",
            "--gate-bias-beta2", "0.999",
            "--gate-bias-weight-decay", "0.02",
        ])

        self.assertEqual(default_args.gate_feature_dim, 16)
        self.assertEqual(default_args.gate_delta_scale, 0.5)
        self.assertIsNone(default_args.gate_warmup_frac)
        self.assertIsNone(default_args.gate_cooldown_frac)
        self.assertEqual(default_args.gate_freeze_steps, 0)
        self.assertEqual(default_args.gate_lr, 0.008)
        self.assertEqual(default_args.gate_beta2, 0.99)
        self.assertEqual(default_args.gate_weight_decay, 0.0)
        self.assertIsNone(default_args.gate_trunk_lr)
        self.assertIsNone(default_args.gate_trunk_beta2)
        self.assertIsNone(default_args.gate_trunk_weight_decay)
        self.assertIsNone(default_args.gate_bias_lr)
        self.assertIsNone(default_args.gate_bias_beta2)
        self.assertIsNone(default_args.gate_bias_weight_decay)
        self.assertEqual(tuned_args.gate_feature_dim, 24)
        self.assertEqual(tuned_args.gate_delta_scale, 0.75)
        self.assertEqual(tuned_args.gate_warmup_frac, 0.1)
        self.assertEqual(tuned_args.gate_cooldown_frac, 0.2)
        self.assertEqual(tuned_args.gate_freeze_steps, 50)
        self.assertEqual(tuned_args.gate_lr, 0.006)
        self.assertEqual(tuned_args.gate_beta2, 0.997)
        self.assertEqual(tuned_args.gate_weight_decay, 0.01)
        self.assertEqual(tuned_args.gate_trunk_lr, 0.002)
        self.assertEqual(tuned_args.gate_trunk_beta2, 0.999)
        self.assertEqual(tuned_args.gate_trunk_weight_decay, 0.03)
        self.assertEqual(tuned_args.gate_bias_lr, 0.002)
        self.assertEqual(tuned_args.gate_bias_beta2, 0.999)
        self.assertEqual(tuned_args.gate_bias_weight_decay, 0.02)

    def test_gate_delta_scale_must_be_in_unit_interval(self) -> None:
        args = build_parser().parse_args(["--learned-residual-gates", "--gate-delta-scale", "0"])

        with self.assertRaisesRegex(ValueError, "gate-delta-scale"):
            validate_args(args)

    def test_gate_bias_tuning_args_are_valid_when_positive(self) -> None:
        args = build_parser().parse_args([
            "--learned-residual-gates",
            "--gate-trunk-lr", "0.001",
            "--gate-trunk-beta2", "0.999",
            "--gate-trunk-weight-decay", "0.0",
            "--gate-bias-lr", "0.001",
            "--gate-bias-beta2", "0.999",
            "--gate-bias-weight-decay", "0.0",
            "--gate-warmup-frac", "0.1",
            "--gate-cooldown-frac", "0.2",
            "--gate-freeze-steps", "10",
        ])

        validate_args(args)

    def test_gate_bias_lr_must_be_positive_when_set(self) -> None:
        args = build_parser().parse_args(["--learned-residual-gates", "--gate-bias-lr", "0"])

        with self.assertRaisesRegex(ValueError, "gate-bias-lr"):
            validate_args(args)

    def test_gate_trunk_lr_must_be_positive_when_set(self) -> None:
        args = build_parser().parse_args(["--learned-residual-gates", "--gate-trunk-lr", "0"])

        with self.assertRaisesRegex(ValueError, "gate-trunk-lr"):
            validate_args(args)

    def test_gate_freeze_steps_must_be_less_than_train_steps(self) -> None:
        args = build_parser().parse_args(["--learned-residual-gates", "--train-steps", "100", "--gate-freeze-steps", "100"])

        with self.assertRaisesRegex(ValueError, "gate-freeze-steps"):
            validate_args(args)

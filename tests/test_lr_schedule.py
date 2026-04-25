from __future__ import annotations

import types
import unittest

from train_gpt_simple import apply_lr_scales
from train_gpt_simple import get_gate_lr_scale, get_lr_scale, resolve_schedule, validate_schedule


class LrScheduleTests(unittest.TestCase):
    def test_returns_constant_scale_without_warmup_before_cooldown(self) -> None:
        self.assertEqual(get_lr_scale(0, train_steps=10, warmup_frac=0.0, cooldown_frac=0.2), 1.0)
        self.assertEqual(get_lr_scale(7, train_steps=10, warmup_frac=0.0, cooldown_frac=0.2), 1.0)

    def test_applies_linear_warmup_then_plateau_then_cooldown(self) -> None:
        self.assertEqual(get_lr_scale(0, train_steps=10, warmup_frac=0.4, cooldown_frac=0.2), 0.25)
        self.assertEqual(get_lr_scale(1, train_steps=10, warmup_frac=0.4, cooldown_frac=0.2), 0.5)
        self.assertEqual(get_lr_scale(3, train_steps=10, warmup_frac=0.4, cooldown_frac=0.2), 1.0)
        self.assertEqual(get_lr_scale(7, train_steps=10, warmup_frac=0.4, cooldown_frac=0.2), 1.0)
        self.assertEqual(get_lr_scale(8, train_steps=10, warmup_frac=0.4, cooldown_frac=0.2), 1.0)
        self.assertEqual(get_lr_scale(9, train_steps=10, warmup_frac=0.4, cooldown_frac=0.2), 0.5)

    def test_resolves_fraction_to_integer_warmup_steps(self) -> None:
        warmup_steps, cooldown_steps = resolve_schedule(train_steps=11, warmup_frac=0.4, cooldown_frac=0.2)
        self.assertEqual((warmup_steps, cooldown_steps), (4, 2))

    def test_rejects_overlapping_warmup_and_cooldown(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not exceed 1"):
            validate_schedule(train_steps=10, warmup_frac=0.5, cooldown_frac=0.6)

    def test_gate_schedule_matches_base_schedule_without_freeze(self) -> None:
        self.assertEqual(
            get_gate_lr_scale(4, train_steps=10, warmup_frac=0.2, cooldown_frac=0.3, freeze_steps=0),
            get_lr_scale(4, train_steps=10, warmup_frac=0.2, cooldown_frac=0.3),
        )

    def test_gate_schedule_freezes_final_steps(self) -> None:
        self.assertEqual(get_gate_lr_scale(6, train_steps=10, warmup_frac=0.0, cooldown_frac=0.5, freeze_steps=2), 0.5)
        self.assertEqual(get_gate_lr_scale(7, train_steps=10, warmup_frac=0.0, cooldown_frac=0.5, freeze_steps=2), 0.25)
        self.assertEqual(get_gate_lr_scale(8, train_steps=10, warmup_frac=0.0, cooldown_frac=0.5, freeze_steps=2), 0.0)
        self.assertEqual(get_gate_lr_scale(9, train_steps=10, warmup_frac=0.0, cooldown_frac=0.5, freeze_steps=2), 0.0)

    def test_apply_lr_scales_only_overrides_gate_groups(self) -> None:
        optimizer_a = types.SimpleNamespace(
            param_groups=[
                {"group_name": "adam_head", "initial_lr": 1.0, "lr": -1.0},
                {"group_name": "gate_trunk", "initial_lr": 2.0, "lr": -1.0},
                {"group_name": "gate_head", "initial_lr": 3.0, "lr": -1.0},
                {"group_name": "gate_bias", "initial_lr": 4.0, "lr": -1.0},
            ]
        )
        optimizer_b = types.SimpleNamespace(param_groups=[{"group_name": "muon_hidden_matrix", "initial_lr": 5.0, "lr": -1.0}])

        current_lrs = apply_lr_scales([optimizer_a, optimizer_b], base_lr_scale=0.5, gate_lr_scale=0.25)

        self.assertEqual(current_lrs["adam_head"], 0.5)
        self.assertEqual(current_lrs["gate_trunk"], 0.5)
        self.assertEqual(current_lrs["gate_head"], 0.75)
        self.assertEqual(current_lrs["gate_bias"], 1.0)
        self.assertEqual(current_lrs["muon_hidden_matrix"], 2.5)

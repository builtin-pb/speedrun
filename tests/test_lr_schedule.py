from __future__ import annotations

import unittest

from train_gpt_simple import get_lr_scale, resolve_schedule, validate_schedule


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

    def test_applies_cooldown_floor_scale(self) -> None:
        self.assertAlmostEqual(
            get_lr_scale(9, train_steps=10, warmup_frac=0.0, cooldown_frac=0.2, cooldown_floor_scale=0.2),
            0.6,
        )

    def test_rejects_invalid_cooldown_floor_scale(self) -> None:
        with self.assertRaisesRegex(ValueError, "cooldown_floor_scale"):
            get_lr_scale(0, train_steps=10, warmup_frac=0.0, cooldown_frac=0.2, cooldown_floor_scale=1.1)

    def test_supports_absolute_cooldown_override(self) -> None:
        self.assertEqual(
            get_lr_scale(4, train_steps=10, warmup_frac=0.0, cooldown_frac=0.2, cooldown_start_step=5, cooldown_steps=3),
            1.0,
        )
        self.assertAlmostEqual(
            get_lr_scale(6, train_steps=10, warmup_frac=0.0, cooldown_frac=0.2, cooldown_start_step=5, cooldown_steps=3),
            2.0 / 3.0,
        )
        self.assertEqual(
            get_lr_scale(8, train_steps=10, warmup_frac=0.0, cooldown_frac=0.2, cooldown_start_step=5, cooldown_steps=3),
            0.0,
        )

    def test_rejects_partial_absolute_cooldown_override(self) -> None:
        with self.assertRaisesRegex(ValueError, "provided together"):
            get_lr_scale(0, train_steps=10, warmup_frac=0.0, cooldown_frac=0.2, cooldown_start_step=5)

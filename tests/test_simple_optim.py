from __future__ import annotations

import unittest

from simple_model import GPT, GPTConfig
from simple_optim import build_optimizers


class BuildOptimizersTests(unittest.TestCase):
    def test_splits_muon_residual_projections_into_their_own_group(self) -> None:
        model = GPT(GPTConfig())

        _, muon = build_optimizers(model, muon_residual_lr_scale=0.7, muon_residual_momentum=0.8, fused_adamw=False)

        self.assertEqual(len(muon.param_groups), 2)
        main_group, residual_group = muon.param_groups
        self.assertEqual(main_group["name"], "muon_main")
        self.assertEqual(residual_group["name"], "muon_residual")
        self.assertAlmostEqual(residual_group["lr"], main_group["lr"] * 0.7)
        self.assertEqual(residual_group["momentum"], 0.8)
        self.assertEqual(len(residual_group["params"]), 24)
        self.assertEqual(len(main_group["params"]), 48)


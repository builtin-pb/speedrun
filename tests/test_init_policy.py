from __future__ import annotations

import unittest

from simple_model import lm_head_init_std, residual_proj_init_scale, spectral_init_std


class InitPolicyTests(unittest.TestCase):
    def test_spectral_init_std_scales_base_formula(self) -> None:
        self.assertAlmostEqual(spectral_init_std(256, 256), 1 / 32)
        self.assertAlmostEqual(spectral_init_std(256, 256, scale=1.5), 1.5 / 32)

    def test_residual_proj_init_scale_divides_by_sqrt_depth(self) -> None:
        self.assertAlmostEqual(residual_proj_init_scale(num_layers=16), 0.25)

    def test_lm_head_init_std_matches_mup_default(self) -> None:
        self.assertAlmostEqual(lm_head_init_std(model_dim=256), 1 / 256)

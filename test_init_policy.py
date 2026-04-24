from __future__ import annotations

import unittest

from init_policy import lm_head_init_std, residual_proj_init_scale, spectral_init_std


class InitPolicyTests(unittest.TestCase):
    def test_spectral_init_std_scales_base_formula(self) -> None:
        self.assertAlmostEqual(spectral_init_std(256, 256), 1 / 32)
        self.assertAlmostEqual(spectral_init_std(256, 256, scale=1.5), 1.5 / 32)

    def test_residual_proj_init_scale_optionally_divides_by_sqrt_depth(self) -> None:
        self.assertEqual(residual_proj_init_scale(1.2, num_layers=12, divide_by_sqrt_depth=False), 1.2)
        self.assertAlmostEqual(
            residual_proj_init_scale(1.2, num_layers=16, divide_by_sqrt_depth=True),
            0.3,
        )

    def test_lm_head_init_std_uses_requested_mode(self) -> None:
        self.assertAlmostEqual(
            lm_head_init_std("mup", model_dim=256, embed_init_std=1.0, mup_lm_head_scale=1.5),
            1.5 / 256,
        )
        self.assertEqual(
            lm_head_init_std("embedding", model_dim=256, embed_init_std=0.7, mup_lm_head_scale=1.5),
            0.7,
        )
        self.assertIsNone(
            lm_head_init_std("torch-default", model_dim=256, embed_init_std=0.7, mup_lm_head_scale=1.5),
        )


if __name__ == "__main__":
    unittest.main()

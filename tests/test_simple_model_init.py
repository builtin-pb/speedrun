from __future__ import annotations

import math
import unittest

import torch

from simple_model import GPT, GPTConfig


def spectral_std(n_in: int, n_out: int, *, scale: float = 1.0) -> float:
    return scale * math.sqrt(n_out / n_in) / (math.sqrt(n_in) + math.sqrt(n_out))


class SimpleModelInitializationTests(unittest.TestCase):
    def test_gpt_uses_fixed_mup_initialization_defaults(self) -> None:
        torch.manual_seed(0)
        config = GPTConfig(
            vocab_size=4096,
            num_layers=4,
            model_dim=256,
            head_dim=64,
            mlp_expansion=4,
        )

        model = GPT(config)

        q_std = float(model.blocks[0].attn.q.weight.float().std(unbiased=False).item())
        attn_proj_std = float(model.blocks[0].attn.proj.weight.float().std(unbiased=False).item())
        fc_std = float(model.blocks[0].mlp.fc.weight.float().std(unbiased=False).item())
        mlp_proj_std = float(model.blocks[0].mlp.proj.weight.float().std(unbiased=False).item())
        lm_head_std = float(model.proj.weight.float().std(unbiased=False).item())
        embed_std = float(model.embed.weight.float().std(unbiased=False).item())

        self.assertAlmostEqual(q_std, spectral_std(256, 256), delta=0.003)
        self.assertAlmostEqual(fc_std, spectral_std(256, 1024), delta=0.003)
        self.assertAlmostEqual(attn_proj_std, spectral_std(256, 256, scale=0.5), delta=0.002)
        self.assertAlmostEqual(mlp_proj_std, spectral_std(1024, 256, scale=0.5), delta=0.002)
        self.assertAlmostEqual(lm_head_std, 1 / config.model_dim, delta=0.0005)
        self.assertAlmostEqual(embed_std, 1.0, delta=0.05)

from __future__ import annotations

import unittest

import torch
from torch import nn
import torch.nn.functional as F

from simple_model import Block, GPTConfig


class _RecordingUpdate(nn.Module):
    def __init__(self, update: torch.Tensor) -> None:
        super().__init__()
        self.update = update
        self.inputs: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor, observer=None, block_idx=None) -> torch.Tensor:
        self.inputs.append(x.detach().clone())
        return self.update.to(device=x.device, dtype=x.dtype).expand_as(x)


class SimpleModelNormPositionTests(unittest.TestCase):
    def test_pre_norm_block_normalizes_inputs_before_residual_updates(self) -> None:
        block = Block(GPTConfig(vocab_size=16, num_layers=1, model_dim=4, head_dim=4))
        attn_update = torch.tensor([[[0.5, -1.0, 1.5, -2.0]]])
        mlp_update = torch.tensor([[[-0.25, 0.75, -1.25, 1.75]]])
        block.attn = _RecordingUpdate(attn_update)
        block.mlp = _RecordingUpdate(mlp_update)
        x = torch.tensor([[[2.0, 4.0, 6.0, 8.0]]])

        y = block(x)

        attn_expected = F.rms_norm(x, (x.size(-1),))
        mlp_expected = F.rms_norm(x + attn_update, (x.size(-1),))
        torch.testing.assert_close(block.attn.inputs[0], attn_expected)
        torch.testing.assert_close(block.mlp.inputs[0], mlp_expected)
        torch.testing.assert_close(y, x + attn_update + mlp_update)

    def test_post_norm_block_normalizes_after_residual_updates(self) -> None:
        block = Block(GPTConfig(vocab_size=16, num_layers=1, model_dim=4, head_dim=4, layer_norm_position="post"))
        attn_update = torch.tensor([[[0.5, -1.0, 1.5, -2.0]]])
        mlp_update = torch.tensor([[[-0.25, 0.75, -1.25, 1.75]]])
        block.attn = _RecordingUpdate(attn_update)
        block.mlp = _RecordingUpdate(mlp_update)
        x = torch.tensor([[[2.0, 4.0, 6.0, 8.0]]])

        y = block(x)

        after_attn = F.rms_norm(x + attn_update, (x.size(-1),))
        torch.testing.assert_close(block.attn.inputs[0], x)
        torch.testing.assert_close(block.mlp.inputs[0], after_attn)
        torch.testing.assert_close(y, F.rms_norm(after_attn + mlp_update, (x.size(-1),)))

    def test_forward_residual_depth_scaling_scales_updates_before_residual_adds(self) -> None:
        block = Block(GPTConfig(vocab_size=16, num_layers=4, model_dim=4, head_dim=4, residual_depth_scale="forward"))
        attn_update = torch.tensor([[[0.5, -1.0, 1.5, -2.0]]])
        mlp_update = torch.tensor([[[-0.25, 0.75, -1.25, 1.75]]])
        block.attn = _RecordingUpdate(attn_update)
        block.mlp = _RecordingUpdate(mlp_update)
        x = torch.tensor([[[2.0, 4.0, 6.0, 8.0]]])

        y = block(x)

        scale = 0.5
        after_attn = x + scale * attn_update
        torch.testing.assert_close(block.mlp.inputs[0], F.rms_norm(after_attn, (x.size(-1),)))
        torch.testing.assert_close(y, after_attn + scale * mlp_update)

    def test_none_residual_depth_scaling_leaves_post_norm_updates_unscaled(self) -> None:
        block = Block(
            GPTConfig(
                vocab_size=16,
                num_layers=4,
                model_dim=4,
                head_dim=4,
                layer_norm_position="post",
                residual_depth_scale="none",
            )
        )
        attn_update = torch.tensor([[[0.5, -1.0, 1.5, -2.0]]])
        mlp_update = torch.tensor([[[-0.25, 0.75, -1.25, 1.75]]])
        block.attn = _RecordingUpdate(attn_update)
        block.mlp = _RecordingUpdate(mlp_update)
        x = torch.tensor([[[2.0, 4.0, 6.0, 8.0]]])

        y = block(x)

        after_attn = F.rms_norm(x + attn_update, (x.size(-1),))
        torch.testing.assert_close(block.mlp.inputs[0], after_attn)
        torch.testing.assert_close(y, F.rms_norm(after_attn + mlp_update, (x.size(-1),)))

    def test_gpt_config_keeps_existing_positional_arguments_compatible(self) -> None:
        config = GPTConfig(16, 1, 4, 4, 1, 2048.0, 0.25, 12.0)

        self.assertEqual(config.rope_base, 2048.0)
        self.assertEqual(config.attention_scale, 0.25)
        self.assertEqual(config.logit_softcap, 12.0)
        self.assertEqual(config.layer_norm_position, "pre")
        self.assertEqual(config.residual_depth_scale, "init")

    def test_gpt_config_rejects_unknown_layer_norm_position(self) -> None:
        with self.assertRaisesRegex(ValueError, "layer_norm_position"):
            GPTConfig(vocab_size=16, num_layers=1, model_dim=4, head_dim=4, layer_norm_position="sandwich")

    def test_gpt_config_rejects_unknown_residual_depth_scale(self) -> None:
        with self.assertRaisesRegex(ValueError, "residual_depth_scale"):
            GPTConfig(vocab_size=16, num_layers=1, model_dim=4, head_dim=4, residual_depth_scale="sometimes")

from __future__ import annotations

import unittest

import torch
from torch import nn
import torch.nn.functional as F

from simple_model import Block, GPT, GPTConfig


class _RecordingUpdate(nn.Module):
    def __init__(self, update: torch.Tensor) -> None:
        super().__init__()
        self.update = update
        self.inputs: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor, observer=None, block_idx=None) -> torch.Tensor:
        self.inputs.append(x.detach().clone())
        return self.update.to(device=x.device, dtype=x.dtype).expand_as(x)


class SimpleModelResidualGateTests(unittest.TestCase):
    def test_dynamic_residual_gates_initialize_to_preserve_baseline_updates(self) -> None:
        block = Block(
            GPTConfig(
                vocab_size=16,
                num_layers=4,
                model_dim=4,
                head_dim=4,
                learned_residual_gates=True,
                gate_feature_dim=2,
            )
        )
        attn_update = torch.tensor([[[0.5, -1.0, 1.5, -2.0]]])
        mlp_update = torch.tensor([[[-0.25, 0.75, -1.25, 1.75]]])
        block.attn = _RecordingUpdate(attn_update)
        block.mlp = _RecordingUpdate(mlp_update)
        x = torch.tensor([[[2.0, 4.0, 6.0, 8.0]]])

        y = block(x)

        self.assertTrue(hasattr(block, "residual_gate_in"))
        self.assertTrue(hasattr(block, "attn_residual_gate"))
        self.assertTrue(hasattr(block, "mlp_residual_gate"))
        torch.testing.assert_close(block.attn_residual_gate.weight.detach(), torch.zeros_like(block.attn_residual_gate.weight))
        torch.testing.assert_close(block.attn_residual_gate.bias.detach(), torch.zeros_like(block.attn_residual_gate.bias))
        torch.testing.assert_close(block.mlp_residual_gate.weight.detach(), torch.zeros_like(block.mlp_residual_gate.weight))
        torch.testing.assert_close(block.mlp_residual_gate.bias.detach(), torch.zeros_like(block.mlp_residual_gate.bias))
        torch.testing.assert_close(block.attn.inputs[0], F.rms_norm(x, (x.size(-1),)))
        torch.testing.assert_close(block.mlp.inputs[0], F.rms_norm(x + attn_update, (x.size(-1),)))
        torch.testing.assert_close(y, x + attn_update + mlp_update)

    def test_dynamic_residual_gates_scale_attention_and_mlp_updates_independently(self) -> None:
        block = Block(
            GPTConfig(
                vocab_size=16,
                num_layers=1,
                model_dim=4,
                head_dim=4,
                learned_residual_gates=True,
                gate_feature_dim=2,
                gate_delta_scale=0.5,
            )
        )
        attn_update = torch.tensor([[[0.5, -1.0, 1.5, -2.0]]])
        mlp_update = torch.tensor([[[-0.25, 0.75, -1.25, 1.75]]])
        block.attn = _RecordingUpdate(attn_update)
        block.mlp = _RecordingUpdate(mlp_update)
        with torch.no_grad():
            block.residual_gate_in.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]))
            block.attn_residual_gate.weight.copy_(torch.tensor([[1.0, -1.0]]))
            block.attn_residual_gate.bias.copy_(torch.tensor([0.1]))
            block.mlp_residual_gate.weight.copy_(torch.tensor([[-0.5, 0.75]]))
            block.mlp_residual_gate.bias.copy_(torch.tensor([-0.2]))
        x = torch.tensor([[[2.0, 4.0, 6.0, 8.0]]])

        y = block(x)

        attn_features = F.linear(F.rms_norm(x, (x.size(-1),)), block.residual_gate_in.weight)
        attn_logits = F.linear(attn_features, block.attn_residual_gate.weight, block.attn_residual_gate.bias)
        attn_gate = 1.0 + 0.5 * torch.tanh(attn_logits)
        after_attn = x + attn_gate * attn_update
        mlp_features = F.linear(F.rms_norm(after_attn, (after_attn.size(-1),)), block.residual_gate_in.weight)
        mlp_logits = F.linear(mlp_features, block.mlp_residual_gate.weight, block.mlp_residual_gate.bias)
        mlp_gate = 1.0 + 0.5 * torch.tanh(mlp_logits)
        torch.testing.assert_close(block.mlp.inputs[0], F.rms_norm(after_attn, (x.size(-1),)))
        torch.testing.assert_close(y, after_attn + mlp_gate * mlp_update)

    def test_dynamic_residual_gates_stay_within_delta_scale_bounds(self) -> None:
        block = Block(
            GPTConfig(
                vocab_size=16,
                num_layers=1,
                model_dim=4,
                head_dim=4,
                learned_residual_gates=True,
                gate_feature_dim=2,
                gate_delta_scale=0.25,
            )
        )
        with torch.no_grad():
            block.residual_gate_in.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]]))
            block.attn_residual_gate.weight.copy_(torch.tensor([[12.0, -12.0]]))
            block.attn_residual_gate.bias.copy_(torch.tensor([8.0]))
        x = torch.tensor([[[2.0, 4.0, 6.0, 8.0]]], dtype=torch.float32)

        gate = block.compute_residual_gate(x, block.attn_residual_gate)

        self.assertTrue(torch.all(gate >= 0.75))
        self.assertTrue(torch.all(gate <= 1.25))

    def test_observer_receives_scaled_residual_updates(self) -> None:
        block = Block(
            GPTConfig(
                vocab_size=16,
                num_layers=1,
                model_dim=4,
                head_dim=4,
                learned_residual_gates=True,
                gate_feature_dim=2,
                gate_delta_scale=1.0,
            )
        )
        block.attn = _RecordingUpdate(torch.tensor([[[2.0, 0.0, 0.0, 0.0]]]))
        block.mlp = _RecordingUpdate(torch.tensor([[[0.0, 4.0, 0.0, 0.0]]]))
        with torch.no_grad():
            block.attn_residual_gate.weight.zero_()
            block.attn_residual_gate.bias.zero_()
            block.mlp_residual_gate.weight.zero_()
            block.mlp_residual_gate.bias.copy_(torch.tensor([torch.atanh(torch.tensor(0.5)).item()]))
        observed: dict[str, torch.Tensor] = {}

        block(
            torch.ones((1, 1, 4)),
            observer=lambda name, tensor: observed.setdefault(name, tensor.detach().clone()),
            block_idx=0,
        )

        torch.testing.assert_close(observed["layer_attn/block_00_update_rms"], torch.tensor([[[2.0, 0.0, 0.0, 0.0]]]))
        torch.testing.assert_close(observed["layer_mlp/block_00_update_rms"], torch.tensor([[[0.0, 6.0, 0.0, 0.0]]]))

    def test_dynamic_residual_gates_receive_gradients(self) -> None:
        block = Block(
            GPTConfig(
                vocab_size=16,
                num_layers=1,
                model_dim=4,
                head_dim=4,
                learned_residual_gates=True,
                gate_feature_dim=2,
            )
        )
        block.attn = _RecordingUpdate(torch.tensor([[[2.0, 0.0, 0.0, 0.0]]]))
        block.mlp = _RecordingUpdate(torch.tensor([[[0.0, 3.0, 0.0, 0.0]]]))

        block(torch.ones((1, 1, 4))).sum().backward()

        self.assertIsNotNone(block.residual_gate_in.weight.grad)
        self.assertIsNotNone(block.attn_residual_gate.weight.grad)
        self.assertIsNotNone(block.mlp_residual_gate.weight.grad)
        self.assertGreater(block.attn_residual_gate.bias.grad.abs().sum().item(), 0.0)
        self.assertGreater(block.mlp_residual_gate.bias.grad.abs().sum().item(), 0.0)

    def test_baseline_does_not_add_learned_residual_gate_parameters(self) -> None:
        block = Block(GPTConfig(vocab_size=16, num_layers=1, model_dim=4, head_dim=4))

        self.assertFalse(hasattr(block, "residual_gate_in"))
        self.assertFalse(hasattr(block, "attn_residual_gate"))
        self.assertFalse(hasattr(block, "mlp_residual_gate"))

    def test_dynamic_gate_parameter_count_stays_below_one_percent(self) -> None:
        config = GPTConfig(vocab_size=128, num_layers=6, model_dim=64, head_dim=32, gate_feature_dim=6)
        baseline = GPT(config)
        gated = GPT(GPTConfig(**{**config.__dict__, "learned_residual_gates": True}))

        baseline_params = sum(param.numel() for param in baseline.parameters())
        gated_params = sum(param.numel() for param in gated.parameters())

        self.assertLess(gated_params - baseline_params, baseline_params * 0.01)

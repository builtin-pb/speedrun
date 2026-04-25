from __future__ import annotations

import unittest

from simple_model import GPT, GPTConfig
from simple_optim import build_optimizers


class SimpleOptimizerTests(unittest.TestCase):
    def test_dynamic_residual_gate_parameters_get_dedicated_adamw_groups(self) -> None:
        model = GPT(
            GPTConfig(
                vocab_size=16,
                num_layers=2,
                model_dim=4,
                head_dim=4,
                learned_residual_gates=True,
                gate_feature_dim=2,
            )
        )

        optimizers = build_optimizers(
            model,
            adam_weight_decay=0.1,
            gate_lr=0.007,
            gate_beta2=0.997,
            gate_weight_decay=0.0,
            gate_trunk_lr=0.003,
            gate_trunk_beta2=0.999,
            gate_trunk_weight_decay=0.02,
            gate_bias_lr=0.002,
            gate_bias_beta2=0.9995,
            gate_bias_weight_decay=0.01,
            fused_adamw=False,
        )

        optimizer_param_ids = [
            id(param)
            for optimizer in optimizers
            for group in optimizer.param_groups
            for param in group["params"]
        ]
        model_param_ids = [id(param) for param in model.parameters() if param.requires_grad]
        adam_groups = {group["group_name"]: group for group in optimizers[0].param_groups}
        trunk_param_ids = {id(block.residual_gate_in.weight) for block in model.blocks}
        head_param_ids = (
            {id(block.attn_residual_gate.weight) for block in model.blocks}
            | {id(block.mlp_residual_gate.weight) for block in model.blocks}
        )
        bias_param_ids = (
            {id(block.attn_residual_gate.bias) for block in model.blocks}
            | {id(block.mlp_residual_gate.bias) for block in model.blocks}
        )

        self.assertEqual(len(optimizer_param_ids), len(set(optimizer_param_ids)))
        self.assertEqual(set(optimizer_param_ids), set(model_param_ids))
        self.assertEqual({id(param) for param in adam_groups["gate_trunk"]["params"]}, trunk_param_ids)
        self.assertEqual(adam_groups["gate_trunk"]["lr"], 0.003)
        self.assertEqual(adam_groups["gate_trunk"]["weight_decay"], 0.02)
        self.assertEqual(adam_groups["gate_trunk"]["betas"], (0.8, 0.999))
        self.assertEqual({id(param) for param in adam_groups["gate_head"]["params"]}, head_param_ids)
        self.assertEqual(adam_groups["gate_head"]["lr"], 0.007)
        self.assertEqual(adam_groups["gate_head"]["weight_decay"], 0.0)
        self.assertEqual(adam_groups["gate_head"]["betas"], (0.8, 0.997))
        self.assertEqual({id(param) for param in adam_groups["gate_bias"]["params"]}, bias_param_ids)
        self.assertEqual(adam_groups["gate_bias"]["lr"], 0.002)
        self.assertEqual(adam_groups["gate_bias"]["weight_decay"], 0.01)
        self.assertEqual(adam_groups["gate_bias"]["betas"], (0.8, 0.9995))
        self.assertEqual(optimizers[1].param_groups[0]["group_name"], "muon_hidden_matrix")

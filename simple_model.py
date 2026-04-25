from dataclasses import dataclass
from collections.abc import Callable
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 50304
    num_layers: int = 12
    model_dim: int = 768
    head_dim: int = 128
    mlp_expansion: int = 4
    rope_base: float = 1024.0
    attention_scale: float = 0.12
    logit_softcap: float = 15.0
    layer_norm_position: str = "pre"
    residual_depth_scale: str = "init"
    learned_residual_gates: bool = False
    gate_feature_dim: int = 16
    gate_delta_scale: float = 0.5

    def __post_init__(self) -> None:
        if self.layer_norm_position not in {"pre", "post"}:
            raise ValueError(f"layer_norm_position must be 'pre' or 'post', got {self.layer_norm_position!r}")
        if self.residual_depth_scale not in {"init", "forward", "none"}:
            raise ValueError(
                f"residual_depth_scale must be 'init', 'forward', or 'none', got {self.residual_depth_scale!r}"
            )
        if not 0.0 < self.gate_delta_scale <= 1.0:
            raise ValueError(f"gate_delta_scale must be in (0, 1], got {self.gate_delta_scale!r}")


def norm(x: Tensor) -> Tensor:
    return F.rms_norm(x, (x.size(-1),))


def spectral_init_std(in_features: int, out_features: int, *, scale: float = 1.0) -> float:
    return scale * math.sqrt(out_features / in_features) / (math.sqrt(in_features) + math.sqrt(out_features))


def residual_proj_init_scale(*, num_layers: int, residual_depth_scale: str = "init") -> float:
    if residual_depth_scale != "init":
        return 1.0
    return 1.0 / math.sqrt(num_layers)


def residual_forward_scale(*, num_layers: int, residual_depth_scale: str) -> float:
    if residual_depth_scale != "forward":
        return 1.0
    return 1.0 / math.sqrt(num_layers)


def lm_head_init_std(*, model_dim: int) -> float:
    return 1.0 / model_dim


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, *, init_scale: float = 1.0):
        super().__init__(in_features, out_features, bias=False)
        nn.init.normal_(self.weight, mean=0.0, std=spectral_init_std(in_features, out_features, scale=init_scale))

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.bfloat16())


class LMHead(Linear):
    def __init__(self, model_dim: int, vocab_size: int):
        super().__init__(model_dim, vocab_size)
        nn.init.normal_(self.weight, mean=0.0, std=lm_head_init_std(model_dim=model_dim))


class Rotary(nn.Module):
    def __init__(self, dim: int, rope_base: float):
        super().__init__()
        if dim % 4 != 0:
            raise ValueError(f"head_dim must be divisible by 4 for rotary embedding, got {dim}")
        angular_freq = (1 / rope_base) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        self.register_buffer(
            "angular_freq",
            torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)]),
            persistent=False,
        )

    def forward(self, x_bthd: Tensor) -> Tensor:
        pos = torch.arange(x_bthd.size(1), dtype=torch.float32, device=x_bthd.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_bthd.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), dim=3).type_as(x_bthd)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        rope_base: float,
        attention_scale: float,
        *,
        proj_init_scale: float,
    ):
        super().__init__()
        if dim % head_dim != 0:
            raise ValueError(f"model_dim must be divisible by head_dim, got {dim=} and {head_dim=}")
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.attention_scale = attention_scale
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim, init_scale=proj_init_scale)
        self.rotary = Rotary(head_dim, rope_base=rope_base)

    def forward(
        self,
        x: Tensor,
        observer: Callable[[str, Tensor], None] | None = None,
        block_idx: int | None = None,
    ) -> Tensor:
        batch_size, seq_len = x.shape[:2]
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        if observer is not None:
            assert block_idx is not None
            observer(f"matrix_attn_q/block_{block_idx:02d}_act_abs", q)
            observer(f"matrix_attn_k/block_{block_idx:02d}_act_abs", k)
            observer(f"matrix_attn_v/block_{block_idx:02d}_act_abs", v)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            scale=self.attention_scale,
            is_causal=True,
        ).transpose(1, 2)
        y = y.contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        y = self.proj(y)
        if observer is not None:
            observer(f"matrix_attn_proj/block_{block_idx:02d}_act_abs", y)
        return y


class MLP(nn.Module):
    def __init__(self, dim: int, expansion: int, *, proj_init_scale: float):
        super().__init__()
        hdim = expansion * dim
        self.fc = Linear(dim, hdim)
        self.proj = Linear(hdim, dim, init_scale=proj_init_scale)

    def forward(
        self,
        x: Tensor,
        observer: Callable[[str, Tensor], None] | None = None,
        block_idx: int | None = None,
    ) -> Tensor:
        x = self.fc(x)
        if observer is not None:
            assert block_idx is not None
            observer(f"matrix_mlp_fc/block_{block_idx:02d}_act_abs", x)
        x = F.relu(x).square()
        x = self.proj(x)
        if observer is not None:
            observer(f"matrix_mlp_proj/block_{block_idx:02d}_act_abs", x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        proj_init_scale = residual_proj_init_scale(
            num_layers=config.num_layers,
            residual_depth_scale=config.residual_depth_scale,
        )
        self.layer_norm_position = config.layer_norm_position
        self.residual_scale = residual_forward_scale(
            num_layers=config.num_layers,
            residual_depth_scale=config.residual_depth_scale,
        )
        if config.learned_residual_gates:
            if config.gate_feature_dim <= 0:
                raise ValueError(f"gate_feature_dim must be positive, got {config.gate_feature_dim}")
            if config.gate_feature_dim > config.model_dim:
                raise ValueError(f"gate_feature_dim must be <= model_dim, got {config.gate_feature_dim} and {config.model_dim}")
            self.residual_gate_in = nn.Linear(config.model_dim, config.gate_feature_dim, bias=False)
            nn.init.normal_(
                self.residual_gate_in.weight,
                mean=0.0,
                std=spectral_init_std(config.model_dim, config.gate_feature_dim),
            )
            self.attn_residual_gate = nn.Linear(config.gate_feature_dim, 1)
            self.mlp_residual_gate = nn.Linear(config.gate_feature_dim, 1)
            nn.init.zeros_(self.attn_residual_gate.weight)
            nn.init.zeros_(self.attn_residual_gate.bias)
            nn.init.zeros_(self.mlp_residual_gate.weight)
            nn.init.zeros_(self.mlp_residual_gate.bias)
            self.gate_feature_dim = config.gate_feature_dim
            self.gate_delta_scale = config.gate_delta_scale
        self.attn = CausalSelfAttention(
            config.model_dim,
            head_dim=config.head_dim,
            rope_base=config.rope_base,
            attention_scale=config.attention_scale,
            proj_init_scale=proj_init_scale,
        )
        self.mlp = MLP(
            config.model_dim,
            expansion=config.mlp_expansion,
            proj_init_scale=proj_init_scale,
        )

    def compute_residual_gate(self, x: Tensor, gate_proj: nn.Linear | None) -> Tensor:
        if gate_proj is None or not hasattr(self, "residual_gate_in"):
            return x.new_ones((*x.shape[:-1], 1))
        gate_features = F.linear(norm(x), self.residual_gate_in.weight.to(dtype=x.dtype))
        gate_logits = F.linear(
            gate_features,
            gate_proj.weight.to(dtype=x.dtype),
            gate_proj.bias.to(dtype=x.dtype),
        )
        return 1.0 + self.gate_delta_scale * torch.tanh(gate_logits)

    def forward(self, x: Tensor, observer: Callable[[str, Tensor], None] | None = None, block_idx: int | None = None) -> Tensor:
        if observer is not None:
            assert block_idx is not None
            observer(f"layer_attn/block_{block_idx:02d}_input_rms", x)
        attn_input = norm(x) if self.layer_norm_position == "pre" else x
        attn_gate = self.compute_residual_gate(x, getattr(self, "attn_residual_gate", None))
        if observer is not None:
            observer(f"gate_coeff_attn/block_{block_idx:02d}", attn_gate)
        attn_out = self.attn(attn_input, observer=observer, block_idx=block_idx)
        attn_update = self.residual_scale * attn_gate * attn_out
        if observer is not None:
            observer(f"layer_attn/block_{block_idx:02d}_update_rms", attn_update)
        x = x + attn_update
        if self.layer_norm_position == "post":
            x = norm(x)
        if observer is not None:
            observer(f"layer_attn/block_{block_idx:02d}_output_rms", x)
            observer(f"layer_mlp/block_{block_idx:02d}_input_rms", x)
        mlp_gate = self.compute_residual_gate(x, getattr(self, "mlp_residual_gate", None))
        if observer is not None:
            observer(f"gate_coeff_mlp/block_{block_idx:02d}", mlp_gate)
        mlp_input = norm(x) if self.layer_norm_position == "pre" else x
        mlp_out = self.mlp(mlp_input, observer=observer, block_idx=block_idx)
        mlp_update = self.residual_scale * mlp_gate * mlp_out
        if observer is not None:
            observer(f"layer_mlp/block_{block_idx:02d}_update_rms", mlp_update)
        x = x + mlp_update
        if self.layer_norm_position == "post":
            x = norm(x)
        if observer is not None:
            observer(f"layer_mlp/block_{block_idx:02d}_output_rms", x)
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.model_dim).bfloat16()
        nn.init.normal_(self.embed.weight, mean=0.0, std=1.0)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.proj = LMHead(config.model_dim, config.vocab_size)

    def compute_raw_logits(self, inputs: Tensor, observer: Callable[[str, Tensor], None] | None = None) -> Tensor:
        x = self.embed(inputs)
        if observer is not None:
            observer("matrix_embed/act_abs", x)
        x = norm(x)
        if observer is not None:
            observer("layer_embed/activation_rms", x)
        for block_idx, block in enumerate(self.blocks):
            x = block(x, observer=observer, block_idx=block_idx)
        if observer is not None:
            observer("layer_final/residual_rms", x)
        return self.proj(norm(x)).float()

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        logits = self.compute_raw_logits(inputs)
        softcap = self.config.logit_softcap
        logits = softcap * logits * torch.rsqrt(logits.square() + softcap**2)
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")

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
    attention_residual: bool = False


def norm(x: Tensor) -> Tensor:
    return F.rms_norm(x, (x.size(-1),))


def spectral_init_std(in_features: int, out_features: int, *, scale: float = 1.0) -> float:
    return scale * math.sqrt(out_features / in_features) / (math.sqrt(in_features) + math.sqrt(out_features))


def residual_proj_init_scale(*, num_layers: int) -> float:
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


class DepthQuery(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))


class FullAttentionResidual(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query = DepthQuery(dim)

    def forward(
        self,
        history: list[Tensor],
        observer: Callable[[str, Tensor], None] | None = None,
        metric_prefix: str | None = None,
    ) -> Tensor:
        if not history:
            raise ValueError("attention residual history must contain at least one state")

        query = self.query.weight
        logits = torch.stack(
            [(norm(state) * query.to(dtype=state.dtype)).sum(dim=-1).float() for state in history],
            dim=0,
        )
        weights = logits.softmax(dim=0)

        mixed = torch.zeros_like(history[-1])
        for weight, state in zip(weights.unbind(dim=0), history):
            mixed.addcmul_(weight.to(dtype=state.dtype).unsqueeze(-1), state)

        if observer is not None:
            assert metric_prefix is not None
            entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=0)
            observer(f"{metric_prefix}_weight_entropy_rms", entropy)
            observer(f"{metric_prefix}_weight_max_rms", weights.amax(dim=0))
            observer(f"{metric_prefix}_current_weight_rms", weights[-1])
            observer(f"{metric_prefix}_embedding_weight_rms", weights[0])
            zero = weights[0].new_zeros(weights[0].shape)
            observer(f"{metric_prefix}_attn_source_weight_rms", weights[1::2].sum(dim=0) if len(history) > 1 else zero)
            observer(f"{metric_prefix}_mlp_source_weight_rms", weights[2::2].sum(dim=0) if len(history) > 2 else zero)
        return mixed


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        proj_init_scale = residual_proj_init_scale(num_layers=config.num_layers)
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
        self.attention_residual = config.attention_residual
        if self.attention_residual:
            self.attn_res = None if layer_idx == 0 else FullAttentionResidual(config.model_dim)
            self.mlp_res = FullAttentionResidual(config.model_dim)

    def _forward_standard(
        self,
        x: Tensor,
        observer: Callable[[str, Tensor], None] | None = None,
        block_idx: int | None = None,
    ) -> Tensor:
        if observer is not None:
            assert block_idx is not None
            observer(f"layer_attn/block_{block_idx:02d}_input_rms", x)
        attn_out = self.attn(norm(x), observer=observer, block_idx=block_idx)
        if observer is not None:
            observer(f"layer_attn/block_{block_idx:02d}_update_rms", attn_out)
        x = x + attn_out
        if observer is not None:
            observer(f"layer_attn/block_{block_idx:02d}_output_rms", x)
            observer(f"layer_mlp/block_{block_idx:02d}_input_rms", x)
        mlp_out = self.mlp(norm(x), observer=observer, block_idx=block_idx)
        if observer is not None:
            observer(f"layer_mlp/block_{block_idx:02d}_update_rms", mlp_out)
        x = x + mlp_out
        if observer is not None:
            observer(f"layer_mlp/block_{block_idx:02d}_output_rms", x)
        return x

    def _forward_attention_residual(
        self,
        x: Tensor,
        history: list[Tensor],
        observer: Callable[[str, Tensor], None] | None = None,
        block_idx: int | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        assert block_idx is not None or observer is None
        expected_history_len = 1 + 2 * self.layer_idx
        if len(history) != expected_history_len:
            raise ValueError(
                f"attention residual history for block {self.layer_idx} must have length {expected_history_len}, got {len(history)}"
            )
        if self.attn_res is None:
            attn_input = history[-1]
        else:
            attn_input = self.attn_res(
                history,
                observer=observer,
                metric_prefix=f"attnres_attn/block_{block_idx:02d}" if observer is not None else None,
            )
        if observer is not None:
            observer(f"layer_attn/block_{block_idx:02d}_input_rms", attn_input)
        attn_out = self.attn(norm(attn_input), observer=observer, block_idx=block_idx)
        if observer is not None:
            observer(f"layer_attn/block_{block_idx:02d}_update_rms", attn_out)
        history = history + [attn_out]
        if observer is not None:
            observer(f"layer_attn/block_{block_idx:02d}_output_rms", attn_input + attn_out)

        mlp_input = self.mlp_res(
            history,
            observer=observer,
            metric_prefix=f"attnres_mlp/block_{block_idx:02d}" if observer is not None else None,
        )
        if observer is not None:
            observer(f"layer_mlp/block_{block_idx:02d}_input_rms", mlp_input)
        mlp_out = self.mlp(norm(mlp_input), observer=observer, block_idx=block_idx)
        if observer is not None:
            observer(f"layer_mlp/block_{block_idx:02d}_update_rms", mlp_out)
        history = history + [mlp_out]
        if observer is not None:
            observer(f"layer_mlp/block_{block_idx:02d}_output_rms", mlp_input + mlp_out)
        return mlp_out, history

    def forward(
        self,
        x: Tensor,
        history: list[Tensor] | None = None,
        observer: Callable[[str, Tensor], None] | None = None,
        block_idx: int | None = None,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        if self.attention_residual:
            if history is None:
                raise ValueError("attention_residual=True requires a residual history")
            return self._forward_attention_residual(x, history, observer=observer, block_idx=block_idx)
        if history is not None:
            raise ValueError("attention_residual=False does not accept a residual history")
        return self._forward_standard(x, observer=observer, block_idx=block_idx)


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.model_dim).bfloat16()
        nn.init.normal_(self.embed.weight, mean=0.0, std=1.0)
        self.blocks = nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.num_layers)])
        if config.attention_residual:
            self.final_res = FullAttentionResidual(config.model_dim)
        self.proj = LMHead(config.model_dim, config.vocab_size)

    def compute_raw_logits(self, inputs: Tensor, observer: Callable[[str, Tensor], None] | None = None) -> Tensor:
        x = self.embed(inputs)
        if observer is not None:
            observer("matrix_embed/act_abs", x)
        x = norm(x)
        if observer is not None:
            observer("layer_embed/activation_rms", x)
        history = [x] if self.config.attention_residual else None
        for block_idx, block in enumerate(self.blocks):
            if history is None:
                x = block(x, observer=observer, block_idx=block_idx)
            else:
                x, history = block(
                    x,
                    history=history,
                    observer=observer,
                    block_idx=block_idx,
                )
        if history is not None:
            x = self.final_res(
                history,
                observer=observer,
                metric_prefix="attnres_final/output" if observer is not None else None,
            )
        if observer is not None:
            observer("layer_final/residual_rms", x)
        return self.proj(norm(x)).float()

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        logits = self.compute_raw_logits(inputs)
        softcap = self.config.logit_softcap
        logits = softcap * logits * torch.rsqrt(logits.square() + softcap**2)
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")

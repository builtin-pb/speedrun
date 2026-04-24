from dataclasses import dataclass
from collections.abc import Callable

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from init_policy import lm_head_init_std, residual_proj_init_scale, spectral_init_std


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
    hidden_init_scale: float = 1.0
    proj_init_div_by_sqrt_depth: bool = False
    embed_init_std: float = 1.0
    lm_head_init: str = "mup"
    lm_head_mup_scale: float = 1.0


def norm(x: Tensor) -> Tensor:
    return F.rms_norm(x, (x.size(-1),))


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, *, init_scale: float = 1.0):
        super().__init__(in_features, out_features, bias=False)
        nn.init.normal_(self.weight, mean=0.0, std=spectral_init_std(in_features, out_features, scale=init_scale))

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.bfloat16())


class LMHead(Linear):
    def __init__(
        self,
        model_dim: int,
        vocab_size: int,
        *,
        hidden_init_scale: float,
        embed_init_std: float,
        init_mode: str,
        mup_scale: float,
    ):
        super().__init__(model_dim, vocab_size, init_scale=hidden_init_scale)
        std = lm_head_init_std(
            init_mode,
            model_dim=model_dim,
            embed_init_std=embed_init_std,
            mup_lm_head_scale=mup_scale,
        )
        if std is None:
            nn.Linear.reset_parameters(self)
        else:
            nn.init.normal_(self.weight, mean=0.0, std=std)


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
        hidden_init_scale: float,
        proj_init_scale: float,
    ):
        super().__init__()
        if dim % head_dim != 0:
            raise ValueError(f"model_dim must be divisible by head_dim, got {dim=} and {head_dim=}")
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.attention_scale = attention_scale
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim, init_scale=hidden_init_scale)
        self.k = Linear(dim, hdim, init_scale=hidden_init_scale)
        self.v = Linear(dim, hdim, init_scale=hidden_init_scale)
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
    def __init__(self, dim: int, expansion: int, *, hidden_init_scale: float, proj_init_scale: float):
        super().__init__()
        hdim = expansion * dim
        self.fc = Linear(dim, hdim, init_scale=hidden_init_scale)
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
            config.hidden_init_scale,
            num_layers=config.num_layers,
            divide_by_sqrt_depth=config.proj_init_div_by_sqrt_depth,
        )
        self.attn = CausalSelfAttention(
            config.model_dim,
            head_dim=config.head_dim,
            rope_base=config.rope_base,
            attention_scale=config.attention_scale,
            hidden_init_scale=config.hidden_init_scale,
            proj_init_scale=proj_init_scale,
        )
        self.mlp = MLP(
            config.model_dim,
            expansion=config.mlp_expansion,
            hidden_init_scale=config.hidden_init_scale,
            proj_init_scale=proj_init_scale,
        )

    def forward(self, x: Tensor, observer: Callable[[str, Tensor], None] | None = None, block_idx: int | None = None) -> Tensor:
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


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.model_dim).bfloat16()
        nn.init.normal_(self.embed.weight, mean=0.0, std=config.embed_init_std)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.proj = LMHead(
            config.model_dim,
            config.vocab_size,
            hidden_init_scale=config.hidden_init_scale,
            embed_init_std=config.embed_init_std,
            init_mode=config.lm_head_init,
            mup_scale=config.lm_head_mup_scale,
        )

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

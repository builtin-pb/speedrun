from dataclasses import dataclass

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


def norm(x: Tensor) -> Tensor:
    return F.rms_norm(x, (x.size(-1),))


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.bfloat16())


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
    def __init__(self, dim: int, head_dim: int, rope_base: float, attention_scale: float):
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
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim, rope_base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len = x.shape[:2]
        q = self.q(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
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
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, expansion: int):
        super().__init__()
        hdim = expansion * dim
        self.fc = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        x = F.relu(x).square()
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attn = CausalSelfAttention(
            config.model_dim,
            head_dim=config.head_dim,
            rope_base=config.rope_base,
            attention_scale=config.attention_scale,
        )
        self.mlp = MLP(config.model_dim, expansion=config.mlp_expansion)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(norm(x))
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.model_dim).bfloat16()
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.proj = Linear(config.model_dim, config.vocab_size)

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        x = norm(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        logits = self.proj(norm(x)).float()
        softcap = self.config.logit_softcap
        logits = softcap * logits * torch.rsqrt(logits.square() + softcap**2)
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")

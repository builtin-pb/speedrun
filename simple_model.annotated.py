"""
Variable-by-variable muP annotation of simple_model.py.

This is an annotated architecture backbone, not a runnable replacement.
Attention residual is treated as always on.

Symbols used in comments:
    d: model_dim
    V: vocab_size
    L: num_layers
    h: head_dim
    H: num_heads = d / h
    r: mlp_expansion
    m: r * d
    K: len(history)
    alpha: attention-residual logit_scale
    S: RMS of a FullAttentionResidual mixed output before the caller's norm
    j: first history length at which a source is available to later mixers

Assumptions:
    1. Width asymptotic includes both d and L. Batch size, sequence length,
       token count, vocabulary size, head_dim, and mlp_expansion are constants
       unless explicitly noted.
    2. Activation values use per-token RMS over the feature dimension.
       Activation gradients and activation deltas use L2 over that dimension.
    3. Hidden matrices use operator L2 norm on the effective math matrix
       A = weight.T because F.linear applies x @ weight.T.
    4. Embeddings are row-local objects. The relevant variable is E[token],
       not the operator norm of the full table.
    5. The LM head is not a hidden matrix. The relevant norm is column RMS of
       the effective W[d,V], equivalently row RMS of stored weight[V,d].
    6. RMSNorm inputs are assumed not to be near zero.
    7. Attention logits are stable only if h is fixed or attention_scale scales
       as Theta(1/sqrt(h)).
    8. Cross entropy is analyzed per token; sum reduction adds a B*T factor if
       token count is not treated constant.
    9. L=num_layers is included in the asymptotic analysis. In the
       attention-residual path, residual projection init scale is 1, so
       attention and MLP branch outputs are width- and depth-scale Theta(1).
    10. K=len(history)=O(L) for depth mixers. The stable-loss annotation chooses
       dense routing with correlated/coherent source scale S=Theta(1): typical
       softmax weights are Theta(1/K), but the mixed value has RMS Theta(1).
       This is the regime in which the stated per-mixer stable-loss products
       close. Sparse routing is called out as an alternative.
    11. Dense independent zero-mean sources would instead give
       S=Theta(1/sqrt(K)). That is not the default stable-loss regime below:
       after the caller's RMSNorm, value-path gradient scales pick up sqrt(K),
       and the annotations must be rechecked with an extra depth assumption or
       a different residual/value scaling.
    12. A history source is consumed by many later mixers when L grows. Full-path
       activation gradients assume dense-route fan-out contributions add
       incoherently across later mixers. For a source first available at history
       length j, the multiplier is
       G_j = (sum_{K>=j} K^-2)^1/2 = O(1/j), hence O(1). Fully aligned
       contributions would add a log(L/j) factor and are not the chosen regime.
    13. For attention-residual query, the intended trained scale is used because
       the code initializes query exactly at zero.

Rule checklist:
    - Stable loss: <Delta x, dL/dx> = Theta(1).
    - Stable value: |Delta x| = Theta(|x|), using intended trained scale for
      zero/small initial variables.
    - Activation: variables passed through nonlinearities have RMS Theta(1).
"""

from __future__ import annotations

from torch import Tensor
import torch
from torch import nn
import torch.nn.functional as F


def norm(x: Tensor) -> Tensor:
    # signature:
    #   x[..., n]
    # chosen norm:
    #   value RMS over n; grad/delta L2 over n
    # scales:
    #   |x| should be RMS Theta(1), L2 Theta(sqrt(n))
    #   |dL/dx| L2 Theta(1/sqrt(n))
    #   lr N/A
    #   |Delta x| RMS Theta(1), L2 Theta(sqrt(n))
    # consistency:
    #   output has RMS Theta(1); Jacobian is width-neutral if input RMS is not
    #   near zero.
    return F.rms_norm(x, (x.size(-1),))


class Linear(nn.Module):
    # parameter:
    #   weight stored [n_out, n_in], effective A: n_in -> n_out
    # chosen norm:
    #   operator L2
    # scales:
    #   |A|_op Theta(sqrt(n_out / n_in))
    #   |dL/dA|_op Theta(sqrt(n_in / n_out)) for rank-one activation gradient
    #   lr Theta(sqrt(n_out / n_in)) for an operator-normalized update
    #   |Delta A|_op Theta(sqrt(n_out / n_in))
    # rule check:
    #   Delta and gradient operator scales multiply to Theta(1), and
    #   x RMS Theta(1) maps to y RMS Theta(1).
    weight: nn.Parameter

    def forward(self, x: Tensor) -> Tensor:
        # x:
        #   shape [..., n_in]
        #   norm value RMS/L2
        #   |x| RMS Theta(1), L2 Theta(sqrt(n_in))
        #   |dL/dx| L2 Theta(1/sqrt(n_in))
        #   lr N/A
        #   |Delta x| RMS Theta(1), L2 Theta(sqrt(n_in))
        y = F.linear(x, self.weight)
        # y:
        #   shape [..., n_out]
        #   norm value RMS/L2
        #   |y| RMS Theta(1), L2 Theta(sqrt(n_out))
        #   |dL/dy| L2 Theta(1/sqrt(n_out))
        #   lr N/A
        #   |Delta y| RMS Theta(1), L2 Theta(sqrt(n_out))
        return y


class LMHead(Linear):
    # parameter:
    #   stored weight [V, d], effective W: d -> V
    # chosen norm:
    #   row RMS of stored weight, i.e. column RMS of effective W
    # scales:
    #   |W_j| row RMS Theta(1/d), row L2 Theta(1/sqrt(d))
    #   |dL/dW_j| row RMS Theta(1), row L2 Theta(sqrt(d)) for active CE rows
    #   lr Theta(1/d) for row/column-normalized update
    #   |Delta W_j| row RMS Theta(1/d), row L2 Theta(1/sqrt(d))
    # rule check:
    #   Delta row L2 1/sqrt(d) times grad row L2 sqrt(d) is Theta(1);
    #   dL/dh = dL/dz @ W.T has L2 Theta(1/sqrt(d)).
    pass


class Rotary(nn.Module):
    angular_freq: Tensor

    def forward(self, x_bthd: Tensor) -> Tensor:
        # x_bthd:
        #   shape [B, T, H, h]
        #   chosen norm per-head RMS/L2 over h
        #   |x| per-head RMS Theta(1), L2 Theta(sqrt(h))
        #   |dL/dx| per-head L2 Theta(sqrt(h)/d) for fixed h in attention
        #   lr N/A
        #   |Delta x| per-head RMS Theta(1)
        pos = torch.arange(x_bthd.size(1), dtype=torch.float32, device=x_bthd.device)
        # pos:
        #   shape [T], nontrainable
        #   chosen norm absolute scalar
        #   |pos| Theta(1) because T is constant
        #   |dL/dpos| N/A, lr N/A, |Delta pos| N/A
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        # angular_freq, theta:
        #   chosen norm absolute scalar
        #   nontrainable bounded scalar fields, Theta(1) for fixed h/T/rope_base;
        #   |dL/d.| N/A, lr N/A, |Delta .| N/A.
        cos, sin = theta.cos(), theta.sin()
        # cos, sin:
        #   chosen norm absolute scalar
        #   |.| bounded O(1); |dL/d.| N/A, lr N/A, |Delta .| N/A.
        x1, x2 = x_bthd.to(dtype=torch.float32).chunk(2, dim=-1)
        # x1, x2:
        #   shape [B, T, H, h/2]
        #   chosen norm per-head RMS/L2 over h/2
        #   |x1|, |x2| RMS Theta(1), L2 Theta(sqrt(h))
        #   |dL/dx1|, |dL/dx2| per-head L2 Theta(sqrt(h)/d) for fixed h
        #   lr N/A
        #   |Delta x1|, |Delta x2| RMS Theta(1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        # y1, y2:
        #   chosen norm per-head RMS/L2 over h/2
        #   |y1|, |y2| RMS Theta(1), L2 Theta(sqrt(h))
        #   |dL/dy1|, |dL/dy2| per-head L2 Theta(sqrt(h)/d) for fixed h
        #   lr N/A
        #   |Delta y1|, |Delta y2| RMS Theta(1)
        #   bounded rotation is scale-preserving.
        y = torch.cat((y1, y2), dim=3).type_as(x_bthd)
        # y:
        #   shape [B, T, H, h]
        #   chosen norm per-head RMS/L2 over h
        #   |y| per-head RMS Theta(1), L2 Theta(sqrt(h))
        #   |dL/dy| per-head L2 Theta(sqrt(h)/d) for fixed h
        #   lr N/A
        #   |Delta y| per-head RMS Theta(1)
        return y


class CausalSelfAttention(nn.Module):
    q: Linear
    k: Linear
    v: Linear
    proj: Linear
    rotary: Rotary
    attention_scale: float
    num_heads: int
    head_dim: int

    # parameters:
    #   q/k/v effective matrices d -> d:
    #       chosen norm operator L2,
    #       operator norm Theta(1), grad op Theta(1), lr Theta(1),
    #       Delta op Theta(1).
    #   proj effective matrix d -> d:
    #       chosen norm operator L2,
    #       operator norm Theta(1) because attention_residual=True makes
    #       residual_proj_init_scale return 1
    #       |dL/dA|_op Theta(1)
    #       lr Theta(1)
    #       |Delta A|_op Theta(1)

    def forward(self, x: Tensor) -> Tensor:
        # module signature:
        #   input x [B, T, d] is already norm(attn_input)
        #   |x| RMS Theta(1), L2 Theta(sqrt(d))
        #   |dL/dx| L2 Theta(1/sqrt(d))
        #   lr N/A
        #   |Delta x| RMS Theta(1), L2 Theta(sqrt(d))
        batch_size, seq_len = x.shape[:2]

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        # q, k, v raw:
        #   shape [B, T, d]
        #   chosen norm RMS/L2 over d
        #   |.| RMS Theta(1), L2 Theta(sqrt(d))
        #   |dL/d.| L2 Theta(1/sqrt(d))
        #   lr N/A
        #   |Delta .| RMS Theta(1)
        # consistency:
        #   follows Linear d -> d with operator-norm Theta(1).

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        # q, k, v viewed:
        #   shape [B, T, H, h]
        #   chosen norm per-head RMS/L2 over h
        #   |.| per-head RMS Theta(1), per-head L2 Theta(sqrt(h))
        #   |dL/d.| per-head L2 Theta(sqrt(h)/d) when h is fixed and
        #       H=d/h heads share the full L2 Theta(1/sqrt(d)) gradient
        #   lr N/A
        #   |Delta .| per-head RMS Theta(1), per-head L2 Theta(sqrt(h))
        #   view is scale-preserving.

        q, k = norm(q), norm(k)
        # q, k normalized:
        #   shape [B, T, H, h]
        #   chosen norm per-head RMS/L2 over h
        #   |q|, |k| per-head RMS Theta(1), per-head L2 Theta(sqrt(h))
        #   |dL/dq|, |dL/dk| per-head L2 Theta(sqrt(h)/d) for fixed h
        #   lr N/A
        #   |Delta q|, |Delta k| per-head RMS Theta(1)
        #   RMSNorm is width-neutral because raw q/k RMS is Theta(1).

        q, k = self.rotary(q), self.rotary(k)
        # q, k rotated:
        #   shape [B, T, H, h]
        #   chosen norm per-head RMS/L2 over h
        #   |q|, |k| per-head RMS Theta(1), L2 Theta(sqrt(h))
        #   |dL/dq|, |dL/dk| per-head L2 Theta(sqrt(h)/d) for fixed h
        #   lr N/A
        #   |Delta q|, |Delta k| per-head RMS Theta(1)
        #   Rotary is norm-preserving.

        y = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            scale=self.attention_scale,
            is_causal=True,
        ).transpose(1, 2)
        # attention scores alpha_attn * q @ k.T:
        #   shape [B, H, T, T]
        #   chosen norm scalar absolute value over constant T
        #   scalar scale Theta(attention_scale * sqrt(h)) typically.
        #   Stable as Theta(1) only when h is fixed or attention_scale is
        #   Theta(1/sqrt(h)).
        #   |dL/dscore| Theta(h/d) = Theta(1/H) for fixed h
        #   lr N/A
        #   |Delta score| Theta(1)
        # attention weights:
        #   shape [B, H, T, T]
        #   chosen norm simplex entries over constant T
        #   entries Theta(1)
        #   |dL/dweight| Theta(h/d) = Theta(1/H) for fixed h, from
        #       dot(dL/dy_head, v_head)
        #   lr N/A
        #   |Delta weight| Theta(1)
        # value mix before concat:
        #   shape [B, H, T, h], then transposed to [B, T, H, h]
        #   chosen norm per-head RMS/L2 over h
        #   per-head RMS Theta(1), per-head L2 Theta(sqrt(h))
        #   |dL/d value_mix| per-head L2 Theta(sqrt(h)/d)
        #   lr N/A
        #   |Delta value_mix| per-head RMS Theta(1)

        y = y.contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        # y concatenated:
        #   shape [B, T, d]
        #   chosen norm RMS/L2 over d
        #   RMS Theta(1), L2 Theta(sqrt(d))
        #   |dL/dy| L2 Theta(1/sqrt(d))
        #   lr N/A
        #   |Delta y| RMS Theta(1)

        y = self.proj(y)
        # y = attn_out:
        #   shape [B, T, d]
        #   chosen norm RMS/L2 over d
        #   RMS Theta(1), L2 Theta(sqrt(d))
        #   |dL/dy| L2 Theta(1/sqrt(d))
        #   lr N/A for activation
        #   |Delta y| RMS Theta(1)
        # consistency:
        #   output is an O(1) history source because attention-residual
        #   residual projections use scale 1.0.
        # caveat:
        #   this is the module-local gradient scale. Once appended to history,
        #   the full activation gradient also includes later depth-mixer fan-out
        #   as specified in assumption 12.
        return y


class MLP(nn.Module):
    fc: Linear
    proj: Linear

    # parameters:
    #   fc effective matrix d -> m:
    #       chosen norm operator L2,
    #       op norm Theta(sqrt(r)), grad op Theta(1/sqrt(r)),
    #       lr Theta(sqrt(r)), Delta op Theta(sqrt(r)).
    #   proj effective matrix m -> d:
    #       chosen norm operator L2,
    #       op norm Theta(1/sqrt(r)) because attention_residual=True makes
    #       residual_proj_init_scale return 1
    #       |dL/dA|_op Theta(sqrt(r))
    #       lr Theta(1/sqrt(r))
    #       Delta op Theta(1/sqrt(r)).

    def forward(self, x: Tensor) -> Tensor:
        # module signature:
        #   input x [B, T, d] is already norm(mlp_input)
        #   chosen norm RMS/L2 over d
        #   |x| RMS Theta(1), L2 Theta(sqrt(d))
        #   |dL/dx| L2 Theta(1/sqrt(d))
        #   lr N/A
        #   |Delta x| RMS Theta(1)
        x = self.fc(x)
        # fc output:
        #   shape [B, T, m]
        #   chosen norm RMS/L2 over m
        #   RMS Theta(1), L2 Theta(sqrt(m))
        #   |dL/dx| L2 Theta(1/sqrt(m))
        #   lr N/A
        #   |Delta x| RMS Theta(1)
        # rule check:
        #   activation rule satisfied before relu-square.

        x = F.relu(x).square()
        # relu_square:
        #   shape [B, T, m]
        #   chosen norm RMS/L2 over m
        #   RMS Theta(1), L2 Theta(sqrt(m)) for stable preactivation
        #   |dL/dx| L2 Theta(1/sqrt(m)) in RMS/typical analysis
        #   lr N/A
        #   |Delta x| RMS Theta(1)
        # caveat:
        #   worst-coordinate ReLU-square derivative growth is not part of this
        #   RMS muP check.

        x = self.proj(x)
        # mlp_out:
        #   shape [B, T, d]
        #   chosen norm RMS/L2 over d
        #   RMS Theta(1), L2 Theta(sqrt(d))
        #   |dL/dx| L2 Theta(1/sqrt(d))
        #   lr N/A for activation
        #   |Delta x| RMS Theta(1)
        # consistency:
        #   output is an O(1) history source because attention-residual
        #   residual projections use scale 1.0.
        # caveat:
        #   this is the module-local gradient scale. Once appended to history,
        #   the full activation gradient also includes later depth-mixer fan-out
        #   as specified in assumption 12.
        return x


class DepthQuery(nn.Module):
    weight: nn.Parameter

    # parameter query:
    #   shape [d]
    #   chosen norm RMS/L2 over d
    #   |query| RMS Theta(1/(alpha*d)), L2 Theta(1/(alpha*sqrt(d)))
    #   |dL/dquery| in the chosen dense S=Theta(1) regime:
    #       typical RMS Theta(alpha/sqrt(K)),
    #       L2 Theta(alpha*sqrt(d/K)).
    #       Sparse selected routing can be RMS Theta(alpha), L2
    #       Theta(alpha*sqrt(d)).
    #   lr Theta(1/(alpha*d)) for Adam-style coordinate-normalized update
    #   |Delta query| RMS Theta(1/(alpha*d)), L2 Theta(1/(alpha*sqrt(d)))
    # rule check:
    #   query scale keeps each routing logit O(1). Dense routing spreads
    #   gradient over K=O(L) sources. The chosen stable-loss regime sets
    #   S=Theta(1), so the typical dense query gradient carries 1/sqrt(K).
    #   The excluded dense-independent S=Theta(1/sqrt(K)) regime would make the
    #   query gradient K-independent but would not close the value-path stable
    #   loss checks without another depth assumption.
    # mode table:
    #   logit_scale none: alpha=1, query RMS Theta(1/d), lr Theta(1/d)
    #   sqrt_dim: alpha=1/sqrt(d), query RMS Theta(1/sqrt(d)),
    #       lr Theta(1/sqrt(d))
    #   dim: alpha=1/d, query RMS Theta(1), lr Theta(1)
    pass


class FullAttentionResidual(nn.Module):
    query: DepthQuery
    logit_scale: float
    normalize_values: bool

    def forward(self, history: list[Tensor]) -> Tensor:
        # module signature:
        #   history length K, each state [B, T, d]
        #   chosen norm is both per-state RMS/L2 over d and aggregate L2 over
        #       the stacked K*d object when checking stable loss in L
        #   every history state has RMS Theta(1), L2 Theta(sqrt(d)) because
        #       attention-residual branch projections now have scale 1
        #   chosen dense S=Theta(1) regime:
        #       per-entry |dL/dstate| L2 Theta(1/(K*sqrt(d)));
        #       aggregate stacked-history gradient L2 Theta(1/(sqrt(K*d))).
        #       Sparse routing gives selected entries L2 Theta(1/sqrt(d)).
        #   lr N/A
        #   |Delta state| RMS Theta(1); for one dense mixer, the matching
        #       route-weighted state update has aggregate effective L2
        #       Theta(sqrt(K*d)) and stable-loss product Theta(1).
        # caveat:
        #   if S=Theta(1/sqrt(K)), these lines acquire a sqrt(K) factor after
        #   the caller's norm and no longer close as written.
        query = self.query.weight
        # query:
        #   shape [d]
        #   chosen norm RMS/L2 over d
        #   |query| RMS Theta(1/(alpha*d)), L2 Theta(1/(alpha*sqrt(d)))
        #   |dL/dquery| chosen dense S=Theta(1) typical RMS
        #       Theta(alpha/sqrt(K)), L2 Theta(alpha*sqrt(d/K));
        #       sparse selected routing can be RMS Theta(alpha), L2
        #       Theta(alpha*sqrt(d)).
        #   lr Theta(1/(alpha*d)) for coordinate-normalized update
        #   |Delta query| RMS Theta(1/(alpha*d))

        logits = torch.stack(
            [(norm(state) * query.to(dtype=state.dtype)).sum(dim=-1).float() for state in history],
            dim=0,
        )
        # norm(state):
        #   shape [B, T, d]
        #   chosen norm RMS/L2 over d
        #   RMS Theta(1), L2 Theta(sqrt(d))
        #   |dL/d norm(state)| L2 Theta(1/sqrt(d))
        #   lr N/A
        #   |Delta norm(state)| RMS Theta(1)
        # prelogits:
        #   shape [K, B, T]
        #   chosen norm absolute scalar per source; aggregate L2 over K
        #   scalar scale Theta(1/alpha)
        #   chosen dense S=Theta(1) |dL/d prelogit_i| Theta(alpha/K);
        #       sparse selected routing can be Theta(alpha)
        #   lr N/A
        #   |Delta prelogit_i| Theta(1/alpha)

        logits = logits * self.logit_scale
        # logits:
        #   shape [K, B, T]
        #   chosen norm absolute scalar per source; aggregate L2 over K
        #   scalar scale Theta(1)
        #   chosen dense S=Theta(1) |dL/dlogit_i| Theta(1/K) when depth
        #       softmax is nonsaturated; sparse selected routing can be Theta(1)
        #   lr N/A
        #   |Delta logit_i| Theta(1)

        weights = logits.softmax(dim=0)
        # weights:
        #   shape [K, B, T]
        #   chosen norm simplex entries over K
        #   dense-routing entries Theta(1/K); at zero query, each is exactly 1/K.
        #       Sparse routing has O(1) selected entries.
        #   |dL/dweight_i| Theta(1) from dot(value_i, dL/dmixed) in the
        #       chosen S=Theta(1) regime
        #   lr N/A
        #   dense-routing |Delta weight_i| Theta(1/K); sparse selected
        #       |Delta weight_i| Theta(1)

        mixed = torch.zeros_like(history[-1])
        # mixed initial zero:
        #   temporary accumulator; final mixed scale below.

        for weight, state in zip(weights.unbind(dim=0), history):
            # weight:
            #   shape [B, T]
            #   chosen norm absolute scalar
            #   dense-routing |weight| Theta(1/K); sparse selected weight O(1)
            #   |dL/dweight| Theta(1) from dot(value, dL/dmixed) in the
            #       chosen S=Theta(1) regime
            #   lr N/A
            #   |Delta weight| Theta(1/K) dense, Theta(1) sparse
            value = norm(state) if self.normalize_values else state
            # value:
            #   if normalize_values=True:
            #       chosen norm RMS/L2 over d,
            #       RMS Theta(1), L2 Theta(sqrt(d)),
            #       chosen dense S=Theta(1) |dL/dvalue| L2
            #       Theta(1/(K*sqrt(d)));
            #       sparse selected value gradient L2 Theta(1/sqrt(d)),
            #       lr N/A,
            #       |Delta value| RMS Theta(1)
            #   if normalize_values=False:
            #       chosen norm RMS/L2 over d,
            #       all history sources have RMS Theta(1) in the current
            #       attention-residual code path.
            #       chosen dense S=Theta(1) |dL/dvalue| L2
            #       Theta(1/(K*sqrt(d)));
            #       sparse selected value gradient L2 Theta(1/sqrt(d)),
            #       lr N/A,
            #       |Delta value| RMS Theta(1).
            mixed.addcmul_(weight.to(dtype=value.dtype).unsqueeze(-1), value)

        # mixed:
        #   shape [B, T, d]
        #   chosen norm RMS/L2 over d
        #   RMS S=Theta(1) in the chosen coherent/correlated dense regime, and
        #       also Theta(1) for sparse selected routing. If the K values were
        #       independent zero-mean sources, dense uniform mixing would instead
        #       give S=Theta(1/sqrt(K)); that regime is excluded from the default
        #       stable-loss checks above.
        #   |dL/dmixed| L2 Theta(1/sqrt(d)) after the caller's norm in the
        #       chosen S=Theta(1) regime
        #   lr N/A
        #   |Delta mixed| RMS Theta(1), L2 Theta(sqrt(d)) in the chosen regime
        # consistency:
        #   callers apply norm(mixed) before attention/MLP/LMHead, so forward
        #   activation signatures stay width-stable. The backward/stable-loss
        #   checks above require the chosen S=Theta(1) regime; the
        #   S=Theta(1/sqrt(K)) regime needs a separate depth analysis.
        return mixed


class Block(nn.Module):
    attn: CausalSelfAttention
    mlp: MLP
    attn_res: FullAttentionResidual | None
    mlp_res: FullAttentionResidual
    layer_idx: int

    def forward(self, x: Tensor, history: list[Tensor]) -> tuple[Tensor, list[Tensor]]:
        # active module signature with attention residual always on:
        #   x [B, T, d] is present in the Python signature, but it is not the
        #   active residual stream. The active state is history.
        #   chosen norm for x RMS/L2 over d
        #   |x| RMS Theta(1), L2 Theta(sqrt(d)); first call also has RMS
        #       Theta(1) from history[0]
        #   |dL/dx| not on the active path except through history; if used as
        #       an activation, L2 Theta(1/sqrt(d))
        #   lr N/A
        #   |Delta x| matches x value scale
        #   expected len(history) = 1 + 2 * layer_idx.
        #   chosen norm for history entries RMS/L2 over d
        #   all history entries RMS Theta(1)
        #   chosen dense S=Theta(1) single-mixer value-path gradient:
        #       |dL/dhistory_i| L2 Theta(1/(K*sqrt(d))); sparse selected
        #       entries Theta(1/sqrt(d)).
        #       Full-path fan-out over later mixers adds the assumption-12
        #       multiplier G_j=O(1), or log(L/j) if fully aligned.
        #   lr N/A
        #   |Delta history_i| matches each entry's value scale

        if self.attn_res is None:
            attn_input = history[-1]
            # attn_input in layer 0:
            #   shape [B, T, d]
            #   chosen norm RMS/L2 over d
            #   |attn_input| RMS Theta(1), L2 Theta(sqrt(d))
            #   |dL/dattn_input| L2 Theta(1/sqrt(d)) locally; as history[0],
            #       later fan-out follows assumption 12.
            #   lr N/A
            #   |Delta attn_input| RMS Theta(1)
        else:
            attn_input = self.attn_res(history)
            # attn_input:
            #   shape [B, T, d]
            #   chosen norm RMS/L2 over d
            #   RMS S=Theta(1), L2 Theta(sqrt(d)) in the chosen regime
            #   |dL/dattn_input| L2 Theta(1/sqrt(d)) after norm(attn_input)
            #   lr N/A
            #   |Delta attn_input| RMS Theta(1)

        attn_out = self.attn(norm(attn_input))
        # norm(attn_input):
        #   chosen norm RMS/L2 over d
        #   |norm(attn_input)| RMS Theta(1), L2 Theta(sqrt(d))
        #   |dL/d norm(attn_input)| L2 Theta(1/sqrt(d))
        #   lr N/A
        #   |Delta norm(attn_input)| RMS Theta(1)
        # attn_out:
        #   chosen norm RMS/L2 over d
        #   RMS Theta(1), L2 Theta(sqrt(d))
        #   |dL/dattn_out| local branch gradient L2 Theta(1/sqrt(d)).
        #       Full activation gradient includes all later history consumers:
        #       dense incoherent fan-out gives G_j/sqrt(d) with G_j=O(1);
        #       fully aligned consumers would add a log(L/j) factor.
        #   lr N/A
        #   |Delta attn_out| RMS Theta(1)
        #   appended as an O(1) history source.

        history = history + [attn_out]
        # history after attention:
        #   len = 2 + 2 * layer_idx
        #   chosen norm per-entry RMS/L2 over d
        #   new entry attn_out has RMS Theta(1).
        #   chosen dense S=Theta(1) single-mixer value-path gradient:
        #       |dL/dhistory_i| L2 Theta(1/(K*sqrt(d))); sparse selected
        #       entries Theta(1/sqrt(d)).
        #       Full-path fan-out follows assumption 12.
        #   lr N/A
        #   |Delta history_i| matches each source scale

        mlp_input = self.mlp_res(history)
        # mlp_input:
        #   shape [B, T, d]
        #   chosen norm RMS/L2 over d
        #   RMS S=Theta(1), L2 Theta(sqrt(d)) in the chosen regime
        #   |dL/dmlp_input| L2 Theta(1/sqrt(d)) after norm(mlp_input)
        #   lr N/A
        #   |Delta mlp_input| RMS Theta(1)

        mlp_out = self.mlp(norm(mlp_input))
        # norm(mlp_input):
        #   chosen norm RMS/L2 over d
        #   |norm(mlp_input)| RMS Theta(1), L2 Theta(sqrt(d))
        #   |dL/d norm(mlp_input)| L2 Theta(1/sqrt(d))
        #   lr N/A
        #   |Delta norm(mlp_input)| RMS Theta(1)
        # mlp_out:
        #   chosen norm RMS/L2 over d
        #   RMS Theta(1), L2 Theta(sqrt(d))
        #   |dL/dmlp_out| local branch gradient L2 Theta(1/sqrt(d)).
        #       Full activation gradient includes all later history consumers:
        #       dense incoherent fan-out gives G_j/sqrt(d) with G_j=O(1);
        #       fully aligned consumers would add a log(L/j) factor.
        #   lr N/A
        #   |Delta mlp_out| RMS Theta(1)
        #   appended as an O(1) history source.

        history = history + [mlp_out]
        # history after MLP:
        #   len = 3 + 2 * layer_idx = 1 + 2 * (layer_idx + 1)
        #   chosen norm per-entry RMS/L2 over d
        #   new entry mlp_out has RMS Theta(1)
        #   chosen dense S=Theta(1) single-mixer value-path gradient:
        #       |dL/dhistory_i| L2 Theta(1/(K*sqrt(d))); sparse selected
        #       entries Theta(1/sqrt(d)).
        #       Full-path fan-out follows assumption 12.
        #   lr N/A
        #   |Delta history_i| matches each source scale
        # consistency:
        #   this satisfies the next block's expected history length.

        return mlp_out, history
        # returned mlp_out:
        #   chosen norm RMS/L2 over d
        #   RMS Theta(1), L2 Theta(sqrt(d))
        #   |dL/dmlp_out| local L2 Theta(1/sqrt(d)); full-path fan-out follows
        #       assumption 12.
        #   lr N/A
        #   |Delta mlp_out| RMS Theta(1)
        # caveat:
        #   returned x is not the next block's active residual stream; next
        #   block uses history.


class GPT(nn.Module):
    embed: nn.Embedding
    blocks: nn.ModuleList
    final_res: FullAttentionResidual
    proj: LMHead
    logit_softcap: float

    def compute_raw_logits(self, inputs: Tensor) -> Tensor:
        # inputs:
        #   integer token ids [B, T]; not a real-valued width object
        #   chosen norm N/A
        #   |inputs| N/A, |dL/dinputs| N/A, lr N/A, |Delta inputs| N/A
        # embedding parameter selected row E[token]:
        #   chosen norm row RMS/L2 over d
        #   |E[token]| RMS Theta(1), L2 Theta(sqrt(d))
        #   |dL/dE[token]| L2 Theta(1/sqrt(d)), RMS Theta(1/d)
        #   lr Theta(1) for row-normalized update
        #   |Delta E[token]| RMS Theta(1), L2 Theta(sqrt(d))
        x = self.embed(inputs)
        # x = embed_raw:
        #   shape [B, T, d]
        #   chosen norm RMS/L2 over d
        #   selected-row RMS Theta(1), L2 Theta(sqrt(d))
        #   |dL/dx| L2 Theta(1/sqrt(d))
        #   lr N/A for activation; embedding parameter LR Theta(1)
        #   |Delta x| RMS Theta(1)

        x = norm(x)
        # x = history[0]:
        #   shape [B, T, d]
        #   chosen norm RMS/L2 over d
        #   RMS Theta(1), L2 Theta(sqrt(d))
        #   |dL/dx| L2 Theta(1/sqrt(d))
        #   lr N/A
        #   |Delta x| RMS Theta(1)
        history = [x]
        # history:
        #   active residual state for all blocks
        #   chosen norm per-entry RMS/L2 over d
        #   |history[0]| RMS Theta(1), L2 Theta(sqrt(d))
        #   |dL/dhistory[0]| local L2 Theta(1/sqrt(d)); full-path fan-out over
        #       L later mixers is bounded by the assumption-12 incoherent dense
        #       accumulation. Fully aligned fan-out would add log L.
        #   lr N/A
        #   |Delta history[0]| RMS Theta(1)

        for block in self.blocks:
            x, history = block(x, history=history)
            # per block:
            #   x is mlp_out with RMS Theta(1) and is not the active
            #   residual stream for the next block.
            #   history length grows by 2 and remains the active state.

        x = self.final_res(history)
        # final_res output:
        #   shape [B, T, d]
        #   chosen norm RMS/L2 over d
        #   consumes final history length K = 1 + 2L = Theta(L)
        #   RMS Theta(1), L2 Theta(sqrt(d)) in the chosen dense coherent or
        #       sparse routing regime. Dense independent-source averaging would
        #       instead be RMS Theta(1/sqrt(K)) and is not the default
        #       stable-loss regime used in this file.
        #   |dL/dx| L2 Theta(1/sqrt(d)) after final norm(x)
        #   lr N/A
        #   |Delta x| RMS Theta(1)

        h = norm(x)
        # h = final hidden:
        #   shape [B, T, d]
        #   chosen norm RMS/L2 over d
        #   RMS Theta(1), L2 Theta(sqrt(d))
        #   |dL/dh| L2 Theta(1/sqrt(d))
        #   lr N/A
        #   |Delta h| RMS Theta(1)
        # consistency:
        #   matches LMHead input assumption.

        raw_logits = self.proj(h).float()
        # raw_logits:
        #   shape [B, T, V]
        #   chosen norm scalar coordinate; optionally L2 over fixed V
        #   coordinate scale at init Theta(1/sqrt(d)); trained useful logits
        #       may be Theta(1)
        #   |dL/draw_logits| coordinate Theta(1) for CE target/spikes
        #   lr N/A
        #   |Delta raw_logits| coordinate Theta(1) under LMHead update
        # consistency:
        #   LMHead row RMS Theta(1/d) maps CE gradient back to hidden gradient
        #   L2 Theta(1/sqrt(d)).
        return raw_logits

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        # module signature:
        #   inputs [B, T] integer token ids:
        #       chosen norm N/A
        #       |inputs| N/A, |dL/dinputs| N/A, lr N/A, |Delta inputs| N/A
        #   targets [B, T] integer token ids:
        #       chosen norm N/A
        #       |targets| N/A, |dL/dtargets| N/A, lr N/A, |Delta targets| N/A
        #   output scalar loss:
        #       chosen norm scalar
        #       |loss| Theta(1) per token, Theta(B*T) with sum reduction
        #       |dL/dloss| 1 by definition, lr N/A, |Delta loss| N/A
        # targets:
        #   integer class ids [B, T]; determine CE gradient only
        #   chosen norm N/A
        #   |targets| N/A, |dL/dtargets| N/A, lr N/A, |Delta targets| N/A
        logits = self.compute_raw_logits(inputs)
        # logits before softcap:
        #   shape [B, T, V]
        #   chosen norm scalar coordinate; optionally L2 over fixed V
        #   coordinate scale trained/useful Theta(1), init random
        #       Theta(1/sqrt(d))
        #   |dL/dlogits| coordinate Theta(1) after CE/softcap chain in the
        #       nonsaturated regime
        #   lr N/A
        #   |Delta logits| coordinate Theta(1)

        softcap = self.logit_softcap
        # softcap:
        #   chosen norm absolute scalar
        #   |softcap| Theta(1); |dL/dsoftcap| N/A, lr N/A, |Delta softcap| N/A.

        logits = softcap * logits * torch.rsqrt(logits.square() + softcap**2)
        # softcapped logits:
        #   shape [B, T, V]
        #   chosen norm scalar coordinate; optionally L2 over fixed V
        #   coordinate scale Theta(1) when raw logits are Theta(1) and
        #       |logits| << softcap; bounded by softcap.
        #   |dL/d softcapped_logits| coordinate Theta(1) from CE
        #   lr N/A
        #   |Delta softcapped_logits| coordinate Theta(1) in nonsaturated regime
        #   derivative factor is (1 + (z/softcap)^2)^(-3/2), width-neutral for
        #   O(1) logits.

        loss = F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")
        # loss:
        #   chosen norm scalar
        #   |loss| Theta(1) per token, Theta(B*T) with sum reduction.
        #   CE gives dL/dlogits = softmax - one_hot, coordinate Theta(1) in
        #   the nonsaturated/non-certain regime.
        #   lr N/A
        #   |Delta loss| N/A
        # final rule check:
        #   all upstream activation and parameter gradient annotations assume
        #   B*T is constant or absorbed as an external multiplier.
        return loss

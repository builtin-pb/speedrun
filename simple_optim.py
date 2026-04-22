import torch
import torch.distributed as dist
from torch import Tensor, nn


def zeropower_via_newtonschulz5(grad: Tensor) -> Tensor:
    assert grad.ndim >= 2
    update = grad.bfloat16()
    if grad.size(-2) > grad.size(-1):
        update = update.mT

    update = update / (update.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for a, b, c in [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]:
        gram = update @ update.mT
        polynomial = b * gram + c * gram @ gram
        update = a * update + polynomial @ update

    if grad.size(-2) > grad.size(-1):
        update = update.mT
    return update


@torch.compile
def muon_update(grad: Tensor, momentum: Tensor, beta: float = 0.95, nesterov: bool = True) -> Tensor:
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


class Muon(torch.optim.Optimizer):
    def __init__(self, params: list[nn.Parameter], lr: float = 0.02, weight_decay: float = 0.0, momentum: float = 0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        assert params and isinstance(params[0], nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self) -> None:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            for base_i in range(len(params))[::world_size]:
                if base_i + rank < len(params):
                    param = params[base_i + rank]
                    state = self.state[param]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(param)
                    update = muon_update(param.grad, state["momentum_buffer"], beta=group["momentum"])
                    param.mul_(1 - group["lr"] * group["weight_decay"])
                    param.add_(update, alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])


def build_optimizers(
    model: nn.Module,
    *,
    adam_head_lr: float = 1 / 320,
    adam_embed_lr: float = 0.3,
    adam_beta1: float = 0.8,
    adam_beta2: float = 0.95,
    adam_eps: float = 1e-10,
    adam_weight_decay: float = 0.0,
    muon_lr: float = 0.02,
    muon_weight_decay: float = 0.01,
    muon_momentum: float = 0.95,
    fused_adamw: bool = True,
) -> list[torch.optim.Optimizer]:
    hidden_matrix_params = [param for param in model.blocks.parameters() if param.ndim >= 2]
    embed_params = list(model.embed.parameters())
    head_params = [model.proj.weight]

    adam_param_groups = [
        dict(params=head_params, lr=adam_head_lr),
        dict(params=embed_params, lr=adam_embed_lr),
    ]
    adamw_kwargs = dict(
        betas=(adam_beta1, adam_beta2),
        eps=adam_eps,
        weight_decay=adam_weight_decay,
    )
    if fused_adamw:
        adamw_kwargs["fused"] = True
    optimizer1 = torch.optim.AdamW(adam_param_groups, **adamw_kwargs)
    optimizer2 = Muon(hidden_matrix_params, lr=muon_lr, weight_decay=muon_weight_decay, momentum=muon_momentum)
    optimizers = [optimizer1, optimizer2]
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
    return optimizers

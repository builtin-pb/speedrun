from __future__ import annotations


def validate_schedule(*, train_steps: int, warmup_frac: float, cooldown_frac: float) -> None:
    if train_steps <= 0:
        raise ValueError("--train-steps must be positive")
    if not 0 <= warmup_frac < 1:
        raise ValueError("--warmup-frac must be in [0, 1)")
    if not 0 < cooldown_frac <= 1:
        raise ValueError("--cooldown-frac must be in (0, 1]")
    if warmup_frac + cooldown_frac > 1:
        raise ValueError("--warmup-frac plus --cooldown-frac must not exceed 1")

def resolve_schedule(*, train_steps: int, warmup_frac: float, cooldown_frac: float) -> tuple[int, int]:
    validate_schedule(
        train_steps=train_steps,
        warmup_frac=warmup_frac,
        cooldown_frac=cooldown_frac,
    )
    warmup_steps = int(train_steps * warmup_frac)
    cooldown_steps = int(train_steps * cooldown_frac)
    return warmup_steps, cooldown_steps


def get_lr_scale(step: int, *, train_steps: int, warmup_frac: float, cooldown_frac: float) -> float:
    warmup_steps, cooldown_steps = resolve_schedule(
        train_steps=train_steps,
        warmup_frac=warmup_frac,
        cooldown_frac=cooldown_frac,
    )
    if not 0 <= step < train_steps:
        raise ValueError("step must be in [0, train_steps)")
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    if cooldown_steps and step >= train_steps - cooldown_steps:
        return (train_steps - step) / cooldown_steps
    return 1.0

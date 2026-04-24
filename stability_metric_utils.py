from __future__ import annotations


def activation_quantile_metric_names(base_name: str) -> dict[str, str]:
    return {
        "p50": f"{base_name}_p50",
        "p90": f"{base_name}_p90",
        "p99": f"{base_name}_p99",
    }


def summarize_logit_extremes(*, top_values: list[float], bottom_values: list[float]) -> dict[str, float]:
    if not top_values or not bottom_values:
        return {}
    return {
        "logits/top1": float(top_values[0]),
        "logits/top5_mean": float(sum(top_values) / len(top_values)),
        "logits/bottom1": float(bottom_values[0]),
        "logits/bottom5_mean": float(sum(bottom_values) / len(bottom_values)),
    }

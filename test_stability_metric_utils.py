from __future__ import annotations

import unittest

from stability_metric_utils import activation_quantile_metric_names, summarize_logit_extremes


class StabilityMetricUtilsTests(unittest.TestCase):
    def test_builds_activation_quantile_metric_names_without_duplicate_abs_suffix(self) -> None:
        names = activation_quantile_metric_names("matrix_attn_q/block_00_act_abs")

        self.assertEqual(names["p50"], "matrix_attn_q/block_00_act_abs_p50")
        self.assertEqual(names["p90"], "matrix_attn_q/block_00_act_abs_p90")
        self.assertEqual(names["p99"], "matrix_attn_q/block_00_act_abs_p99")

    def test_summarizes_top_and_bottom_logits_using_mean_of_five(self) -> None:
        metrics = summarize_logit_extremes(
            top_values=[9.0, 8.0, 7.0, 6.0, 5.0],
            bottom_values=[-9.0, -8.0, -7.0, -6.0, -5.0],
        )

        self.assertEqual(metrics["logits/top1"], 9.0)
        self.assertEqual(metrics["logits/top5_mean"], 7.0)
        self.assertEqual(metrics["logits/bottom1"], -9.0)
        self.assertEqual(metrics["logits/bottom5_mean"], -7.0)


if __name__ == "__main__":
    unittest.main()

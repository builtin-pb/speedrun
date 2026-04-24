from __future__ import annotations

import unittest
import sys
import types
from contextlib import redirect_stderr
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

from train_gpt_simple import build_gpt_config, build_parser


class TrainArgsTests(unittest.TestCase):
    def test_layer_norm_position_defaults_to_pre_norm(self) -> None:
        args = build_parser().parse_args([])

        self.assertEqual(args.layer_norm_position, "pre")

    def test_layer_norm_position_accepts_post_norm(self) -> None:
        args = build_parser().parse_args(["--layer-norm-position", "post"])

        self.assertEqual(args.layer_norm_position, "post")

    def test_layer_norm_position_rejects_unknown_values(self) -> None:
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["--layer-norm-position", "sandwich"])

    def test_residual_depth_scale_defaults_to_init(self) -> None:
        args = build_parser().parse_args([])

        self.assertEqual(args.residual_depth_scale, "init")

    def test_residual_depth_scale_accepts_none_and_forward(self) -> None:
        for mode in ("none", "forward"):
            with self.subTest(mode=mode):
                args = build_parser().parse_args(["--residual-depth-scale", mode])

                self.assertEqual(args.residual_depth_scale, mode)

    def test_residual_depth_scale_rejects_unknown_values(self) -> None:
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["--residual-depth-scale", "sometimes"])

    def test_build_gpt_config_passes_norm_and_residual_scale_knobs(self) -> None:
        args = build_parser().parse_args(["--layer-norm-position", "post", "--residual-depth-scale", "none"])
        fake_simple_model = types.ModuleType("simple_model")
        fake_simple_model.GPTConfig = lambda **kwargs: SimpleNamespace(**kwargs)

        with patch.dict(sys.modules, {"simple_model": fake_simple_model}):
            config = build_gpt_config(args)

        self.assertEqual(config.layer_norm_position, "post")
        self.assertEqual(config.residual_depth_scale, "none")

import unittest

from tiny_invoker.cli import (
    benchmark_metric_stats,
    build_bench_gpt_neo_parser,
    build_bench_qwen2_parser,
    build_compare_gpt_neo_parser,
    build_compare_qwen2_parser,
    build_convert_safetensors_parser,
    build_generate_qwen2_parser,
)


class CliTest(unittest.TestCase):
    def test_bench_gpt_neo_parser_defaults(self) -> None:
        parser = build_bench_gpt_neo_parser()

        args = parser.parse_args(["roneneldan/TinyStories-33M", "Once upon a time"])

        self.assertEqual(args.model_id, "roneneldan/TinyStories-33M")
        self.assertEqual(args.prompt, "Once upon a time")
        self.assertEqual(args.max_new_tokens, 128)
        self.assertEqual(args.temperature, 0.0)
        self.assertEqual(args.top_k, 20)
        self.assertEqual(args.repeats, 2)
        self.assertEqual(args.warmups, 1)
        self.assertFalse(args.profile)
        self.assertFalse(args.json)

    def test_bench_gpt_neo_parser_accepts_profile_and_json(self) -> None:
        parser = build_bench_gpt_neo_parser()

        args = parser.parse_args(
            ["roneneldan/TinyStories-33M", "Once upon a time", "--profile", "--json"]
        )

        self.assertTrue(args.profile)
        self.assertTrue(args.json)

    def test_bench_qwen2_parser_defaults(self) -> None:
        parser = build_bench_qwen2_parser()

        args = parser.parse_args(["Qwen/Qwen2.5-0.5B", "Hello"])

        self.assertEqual(args.model_id, "Qwen/Qwen2.5-0.5B")
        self.assertEqual(args.prompt, "Hello")
        self.assertEqual(args.weights_file, "model.npz")
        self.assertEqual(args.max_new_tokens, 128)
        self.assertEqual(args.temperature, 0.0)
        self.assertEqual(args.top_k, 20)
        self.assertFalse(args.profile)
        self.assertFalse(args.json)

    def test_benchmark_metric_stats(self) -> None:
        stats = benchmark_metric_stats(
            "prefill_ms",
            [{"prefill_ms": 1.0}, {"prefill_ms": 3.0}],
        )

        self.assertEqual(stats["avg"], 2.0)
        self.assertAlmostEqual(stats["stdev"], 2**0.5)

    def test_compare_gpt_neo_parser_defaults(self) -> None:
        parser = build_compare_gpt_neo_parser()

        args = parser.parse_args(["roneneldan/TinyStories-33M", "Once upon a time"])

        self.assertEqual(args.model_id, "roneneldan/TinyStories-33M")
        self.assertEqual(args.prompt, "Once upon a time")
        self.assertEqual(args.top_k, 10)
        self.assertEqual(args.tolerance, 1.0e-3)
        self.assertFalse(args.fail_on_mismatch)

    def test_compare_gpt_neo_parser_accepts_failure_mode(self) -> None:
        parser = build_compare_gpt_neo_parser()

        args = parser.parse_args(
            [
                "roneneldan/TinyStories-33M",
                "Once upon a time",
                "--top-k",
                "5",
                "--tolerance",
                "0.01",
                "--fail-on-mismatch",
            ]
        )

        self.assertEqual(args.top_k, 5)
        self.assertEqual(args.tolerance, 0.01)
        self.assertTrue(args.fail_on_mismatch)

    def test_convert_safetensors_parser_defaults(self) -> None:
        parser = build_convert_safetensors_parser()

        args = parser.parse_args(["Qwen/Qwen2.5-0.5B"])

        self.assertEqual(args.model_id, "Qwen/Qwen2.5-0.5B")
        self.assertEqual(args.filename, "model.safetensors")
        self.assertFalse(args.uncompressed)

    def test_generate_qwen2_parser_defaults(self) -> None:
        parser = build_generate_qwen2_parser()

        args = parser.parse_args(["Qwen/Qwen2.5-0.5B", "Hello"])

        self.assertEqual(args.model_id, "Qwen/Qwen2.5-0.5B")
        self.assertEqual(args.prompt, "Hello")
        self.assertEqual(args.weights_file, "model.npz")
        self.assertEqual(args.max_new_tokens, 20)

    def test_compare_qwen2_parser_accepts_failure_mode(self) -> None:
        parser = build_compare_qwen2_parser()

        args = parser.parse_args(
            [
                "Qwen/Qwen2.5-0.5B",
                "Hello",
                "--safetensors-file",
                "model-00001-of-00002.safetensors",
                "--fail-on-mismatch",
            ]
        )

        self.assertEqual(args.safetensors_file, "model-00001-of-00002.safetensors")
        self.assertTrue(args.fail_on_mismatch)


if __name__ == "__main__":
    unittest.main()

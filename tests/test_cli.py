import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from tiny_invoker.cli import (
    append_jsonl,
    benchmark_payload_metrics,
    benchmark_metric_stats,
    build_bench_gpt_neo_parser,
    build_bench_qwen2_parser,
    build_bench_server_parser,
    build_compare_bench_parser,
    build_compare_gpt_neo_parser,
    build_compare_qwen2_parser,
    build_convert_safetensors_parser,
    build_generate_qwen2_parser,
    distribution_stats,
    load_benchmark_jsonl,
    run_compare_bench,
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
        self.assertIsNone(args.json_output)

    def test_bench_gpt_neo_parser_accepts_profile_and_json(self) -> None:
        parser = build_bench_gpt_neo_parser()

        args = parser.parse_args(
            [
                "roneneldan/TinyStories-33M",
                "Once upon a time",
                "--profile",
                "--json",
                "--json-output",
                "benchmarks/baseline/gpt.jsonl",
            ]
        )

        self.assertTrue(args.profile)
        self.assertTrue(args.json)
        self.assertEqual(args.json_output, Path("benchmarks/baseline/gpt.jsonl"))

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

    def test_compare_bench_parser_defaults(self) -> None:
        parser = build_compare_bench_parser()

        args = parser.parse_args(["baseline.jsonl", "candidate.jsonl"])

        self.assertEqual(args.baseline, Path("baseline.jsonl"))
        self.assertEqual(args.candidate, Path("candidate.jsonl"))
        self.assertIn("ttft_ms", args.metrics)
        self.assertEqual(args.stat, "avg")
        self.assertFalse(args.all_records)

    def test_bench_server_parser_defaults(self) -> None:
        parser = build_bench_server_parser()

        args = parser.parse_args([])

        self.assertEqual(args.url, "http://127.0.0.1:8000/generate")
        self.assertEqual(args.max_new_tokens, 16)
        self.assertEqual(args.requests, 8)
        self.assertEqual(args.concurrency, 2)
        self.assertFalse(args.json)

    def test_benchmark_metric_stats(self) -> None:
        stats = benchmark_metric_stats(
            "prefill_ms",
            [{"prefill_ms": 1.0}, {"prefill_ms": 3.0}],
        )

        self.assertEqual(stats["avg"], 2.0)
        self.assertAlmostEqual(stats["stdev"], 2**0.5)
        self.assertEqual(stats["p50"], 2.0)
        self.assertAlmostEqual(stats["p95"], 2.9)

    def test_benchmark_payload_metrics_selects_stat(self) -> None:
        payload = {
            "metrics": {
                "ttft_ms": {
                    "avg": 10.0,
                    "p50": 9.0,
                    "p95": 15.0,
                }
            }
        }

        metrics = benchmark_payload_metrics(payload, stat_name="p50")

        self.assertEqual(metrics["ttft_ms"], 9.0)

    def test_benchmark_jsonl_round_trip_skips_text_lines(self) -> None:
        payload = {
            "benchmark_name": "test",
            "metrics": {
                "ttft_ms": {
                    "avg": 1.0,
                    "stdev": 0.0,
                }
            },
        }
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "bench.jsonl"
            path.write_text("human output\n", encoding="utf-8")
            append_jsonl(path, payload)

            records = load_benchmark_jsonl(path)

        self.assertEqual(records, [payload])

    def test_compare_bench_all_records_matches_cases(self) -> None:
        baseline_one = {
            "benchmark_name": "qwen2_runtime",
            "model_id": "Qwen/Qwen2.5-0.5B",
            "prompt": "Hello",
            "prompt_tokens": 1,
            "max_new_tokens": 8,
            "temperature": 0.0,
            "top_k": 20,
            "metrics": {
                "ttft_ms": {"avg": 10.0, "p50": 9.0, "p95": 14.0, "stdev": 1.0}
            },
        }
        baseline_two = {
            "benchmark_name": "qwen2_runtime",
            "model_id": "Qwen/Qwen2.5-0.5B",
            "prompt": "Explain KV cache.",
            "prompt_tokens": 4,
            "max_new_tokens": 16,
            "temperature": 0.0,
            "top_k": 20,
            "metrics": {
                "ttft_ms": {"avg": 20.0, "p50": 19.0, "p95": 24.0, "stdev": 2.0}
            },
        }
        candidate_one = {
            **baseline_one,
            "metrics": {"ttft_ms": {"avg": 8.0, "p50": 7.0, "p95": 12.0, "stdev": 1.5}},
        }
        candidate_two = {
            **baseline_two,
            "metrics": {"ttft_ms": {"avg": 18.0, "p50": 17.0, "p95": 22.0, "stdev": 2.5}},
        }
        with TemporaryDirectory() as tmp_dir:
            baseline_path = Path(tmp_dir) / "baseline.jsonl"
            candidate_path = Path(tmp_dir) / "candidate.jsonl"
            append_jsonl(baseline_path, baseline_one)
            append_jsonl(baseline_path, baseline_two)
            append_jsonl(candidate_path, candidate_one)
            append_jsonl(candidate_path, candidate_two)

            output = StringIO()
            with redirect_stdout(output):
                exit_code = run_compare_bench(
                    [
                        str(baseline_path),
                        str(candidate_path),
                        "--metrics",
                        "ttft_ms",
                        "--stat",
                        "p50",
                        "--all-records",
                    ]
                )

        self.assertEqual(exit_code, 0)
        text = output.getvalue()
        self.assertIn("comparison_mode: all_records", text)
        self.assertIn("matched_cases: 2", text)
        self.assertIn("baseline_p50 candidate_p50", text)
        self.assertIn("Hello", text)
        self.assertIn("Explain KV cache.", text)

    def test_distribution_stats(self) -> None:
        stats = distribution_stats([1.0, 2.0, 3.0, 4.0])

        self.assertEqual(stats["avg"], 2.5)
        self.assertEqual(stats["p50"], 2.5)
        self.assertAlmostEqual(stats["p95"], 3.85)

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

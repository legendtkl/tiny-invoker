import unittest

from tiny_invoker.cli import build_bench_gpt_neo_parser, build_compare_gpt_neo_parser


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

    def test_bench_gpt_neo_parser_accepts_profile(self) -> None:
        parser = build_bench_gpt_neo_parser()

        args = parser.parse_args(["roneneldan/TinyStories-33M", "Once upon a time", "--profile"])

        self.assertTrue(args.profile)

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


if __name__ == "__main__":
    unittest.main()

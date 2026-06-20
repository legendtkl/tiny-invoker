from __future__ import annotations

import argparse
import sys

from tiny_invoker.demo import build_demo_engine
from tiny_invoker.engine import GenerationConfig
from tiny_invoker.hf import fetch_model_info


def build_generate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the tiny learning-oriented inference engine.",
    )
    parser.add_argument("prompt", nargs="?", default="tiny", help="Text prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--trace", action="store_true", help="Print per-step token candidates.")
    return parser


def build_inspect_model_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tiny-invoker inspect-model",
        description="Inspect a Hugging Face causal language model repository.",
    )
    parser.add_argument("model_id", help="Hugging Face model id, for example roneneldan/TinyStories-33M.")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--endpoint", default="https://huggingface.co")
    return parser


def run_generate(argv: list[str]) -> int:
    parser = build_generate_parser()
    args = parser.parse_args(argv)

    engine = build_demo_engine()
    top_k = args.top_k if args.top_k > 0 else None
    result = engine.generate(
        args.prompt,
        config=GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=top_k,
            seed=args.seed,
            trace=args.trace,
        ),
    )

    print(result.text)
    if args.trace:
        print()
        print("trace:")
        for step in result.steps:
            candidates = ", ".join(
                f"{token!r}:{probability:.2f}"
                for token, probability in step.candidates
            )
            print(
                f"{step.step:02d}. prev={step.previous_token!r} "
                f"next={step.chosen_token!r} candidates=[{candidates}]"
            )
    return 0


def run_inspect_model(argv: list[str]) -> int:
    parser = build_inspect_model_parser()
    args = parser.parse_args(argv)

    info = fetch_model_info(
        args.model_id,
        endpoint=args.endpoint,
        revision=args.revision,
    )
    for line in info.summary_lines():
        print(line)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "inspect-model":
        return run_inspect_model(args[1:])
    return run_generate(args)

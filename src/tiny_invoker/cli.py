from __future__ import annotations

import argparse

from tiny_invoker.demo import build_demo_engine
from tiny_invoker.engine import GenerationConfig


def build_parser() -> argparse.ArgumentParser:
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
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

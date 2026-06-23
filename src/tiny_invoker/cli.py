from __future__ import annotations

import argparse
from collections.abc import Callable
import json
import random
from pathlib import Path
from statistics import mean, stdev
import sys
import time
from typing import Any

from tiny_invoker.demo import build_demo_engine
from tiny_invoker.engine import GenerationConfig, InferenceEngine
from tiny_invoker.gpt_neo import NumpyGptNeoLanguageModel
from tiny_invoker.hf import download_model_file, fetch_model_info, model_cache_dir
from tiny_invoker.interfaces import ForwardInput, ForwardMode
from tiny_invoker.qwen2 import NumpyQwen2LanguageModel
from tiny_invoker.sampler import choose_token
from tiny_invoker.server import serve
from tiny_invoker.tokenizer import HfTokenizer
from tiny_invoker.transformer import require_numpy
from tiny_invoker.weights import (
    convert_safetensors_weights_to_npz,
    convert_torch_weights_to_npz,
    load_safetensors_weight_manifest,
    load_torch_weight_manifest,
)


PROFILE_METRIC_NAMES = (
    "embedding_ms",
    "blocks_ms",
    "attention_ms",
    "mlp_ms",
    "final_norm_ms",
    "lm_head_ms",
)

SEGMENTED_BENCHMARK_METRIC_NAMES = (
    "prefill_ms",
    "prefill_tokens_per_second",
    "ttft_ms",
    "decode_forward_ms",
    "decode_forward_total_ms",
    "sampler_with_mask_ms",
    "sampler_total_ms",
    "tpot_ms",
    "decode_tokens_per_second",
    "model_decode_tokens_per_second",
)

END_TO_END_BENCHMARK_METRIC_NAMES = (
    "end_to_end_ms",
    "end_to_end_tokens_per_second",
)

BenchmarkValue = float | int | str
BenchmarkRow = dict[str, BenchmarkValue]
ModelLoader = Callable[[argparse.Namespace], tuple[Any, HfTokenizer, dict[str, Path]]]


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


def build_tokenize_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tiny-invoker tokenize",
        description="Tokenize text with a Hugging Face tokenizer.json file.",
    )
    parser.add_argument("model_id", help="Hugging Face model id, for example roneneldan/TinyStories-33M.")
    parser.add_argument("text", help="Text to encode.")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser


def build_inspect_weights_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tiny-invoker inspect-weights",
        description="Inspect a Hugging Face PyTorch state_dict weight file.",
    )
    parser.add_argument("model_id", help="Hugging Face model id, for example roneneldan/TinyStories-33M.")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--filename", default="pytorch_model.bin")
    parser.add_argument("--limit", type=int, default=80, help="Maximum tensor lines to print. Use 0 for all.")
    return parser


def build_convert_weights_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tiny-invoker convert-weights",
        description="Convert a Hugging Face PyTorch state_dict to a NumPy .npz file.",
    )
    parser.add_argument("model_id", help="Hugging Face model id, for example roneneldan/TinyStories-33M.")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--filename", default="pytorch_model.bin")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--uncompressed", action="store_true", help="Use np.savez instead of np.savez_compressed.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum converted tensor lines to print. Use 0 for all.")
    return parser


def build_inspect_safetensors_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tiny-invoker inspect-safetensors",
        description="Inspect a Hugging Face safetensors weight file.",
    )
    parser.add_argument("model_id", help="Hugging Face model id, for example Qwen/Qwen2.5-0.5B.")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--filename", default="model.safetensors")
    parser.add_argument("--limit", type=int, default=80, help="Maximum tensor lines to print. Use 0 for all.")
    return parser


def build_convert_safetensors_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tiny-invoker convert-safetensors",
        description="Convert a Hugging Face safetensors state_dict to a NumPy .npz file.",
    )
    parser.add_argument("model_id", help="Hugging Face model id, for example Qwen/Qwen2.5-0.5B.")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--filename", default="model.safetensors")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--uncompressed", action="store_true", help="Use np.savez instead of np.savez_compressed.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum converted tensor lines to print. Use 0 for all.")
    return parser


def build_probe_gpt_neo_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tiny-invoker probe-gpt-neo",
        description="Run the NumPy GPT-Neo runtime skeleton and print logits metadata.",
    )
    parser.add_argument("model_id", help="Hugging Face model id, for example roneneldan/TinyStories-33M.")
    parser.add_argument("prompt", help="Prompt text to tokenize and run through the runtime skeleton.")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--weights-file", default="pytorch_model.npz")
    parser.add_argument("--weights-path", type=Path, default=None)
    parser.add_argument("--config-file", default="config.json")
    return parser


def build_generate_gpt_neo_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tiny-invoker generate-gpt-neo",
        description="Generate text with the NumPy GPT-Neo runtime.",
    )
    parser.add_argument("model_id", help="Hugging Face model id, for example roneneldan/TinyStories-33M.")
    parser.add_argument("prompt", help="Prompt text.")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--weights-file", default="pytorch_model.npz")
    parser.add_argument("--weights-path", type=Path, default=None)
    parser.add_argument("--config-file", default="config.json")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--trace", action="store_true")
    return parser


def build_serve_gpt_neo_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tiny-invoker serve-gpt-neo",
        description="Serve the NumPy GPT-Neo runtime over a tiny local HTTP API.",
    )
    parser.add_argument("model_id", help="Hugging Face model id, for example roneneldan/TinyStories-33M.")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--weights-file", default="pytorch_model.npz")
    parser.add_argument("--weights-path", type=Path, default=None)
    parser.add_argument("--config-file", default="config.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def add_benchmark_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--sample-chars", type=int, default=500)
    parser.add_argument("--profile", action="store_true", help="Print internal prefill/decode forward timing.")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable JSON summary after text output.")


def build_bench_gpt_neo_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tiny-invoker bench-gpt-neo",
        description="Benchmark the NumPy GPT-Neo runtime with segmented timing.",
    )
    parser.add_argument("model_id", help="Hugging Face model id, for example roneneldan/TinyStories-33M.")
    parser.add_argument("prompt", help="Prompt text.")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--weights-file", default="pytorch_model.npz")
    parser.add_argument("--weights-path", type=Path, default=None)
    parser.add_argument("--config-file", default="config.json")
    add_benchmark_arguments(parser)
    return parser


def build_bench_qwen2_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tiny-invoker bench-qwen2",
        description="Benchmark the NumPy Qwen2 runtime with segmented timing.",
    )
    parser.add_argument("model_id", help="Hugging Face model id, for example Qwen/Qwen2.5-0.5B.")
    parser.add_argument("prompt", help="Prompt text.")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--weights-file", default="model.npz")
    parser.add_argument("--weights-path", type=Path, default=None)
    parser.add_argument("--config-file", default="config.json")
    add_benchmark_arguments(parser)
    return parser


def build_compare_gpt_neo_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tiny-invoker compare-gpt-neo",
        description="Compare NumPy GPT-Neo logits against Hugging Face Transformers.",
    )
    parser.add_argument("model_id", help="Hugging Face model id, for example roneneldan/TinyStories-33M.")
    parser.add_argument("prompt", help="Prompt text.")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--weights-file", default="pytorch_model.npz")
    parser.add_argument("--weights-path", type=Path, default=None)
    parser.add_argument("--config-file", default="config.json")
    parser.add_argument("--torch-weights-file", default="pytorch_model.bin")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=1.0e-3)
    parser.add_argument("--fail-on-mismatch", action="store_true")
    return parser


def build_probe_qwen2_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tiny-invoker probe-qwen2",
        description="Run the NumPy Qwen2 runtime and print logits metadata.",
    )
    parser.add_argument("model_id", help="Hugging Face model id, for example Qwen/Qwen2.5-0.5B.")
    parser.add_argument("prompt", help="Prompt text to tokenize and run through the runtime.")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--weights-file", default="model.npz")
    parser.add_argument("--weights-path", type=Path, default=None)
    parser.add_argument("--config-file", default="config.json")
    return parser


def build_generate_qwen2_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tiny-invoker generate-qwen2",
        description="Generate text with the NumPy Qwen2 runtime.",
    )
    parser.add_argument("model_id", help="Hugging Face model id, for example Qwen/Qwen2.5-0.5B.")
    parser.add_argument("prompt", help="Prompt text.")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--weights-file", default="model.npz")
    parser.add_argument("--weights-path", type=Path, default=None)
    parser.add_argument("--config-file", default="config.json")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--trace", action="store_true")
    return parser


def build_compare_qwen2_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tiny-invoker compare-qwen2",
        description="Compare NumPy Qwen2 logits against Hugging Face Transformers.",
    )
    parser.add_argument("model_id", help="Hugging Face model id, for example Qwen/Qwen2.5-0.5B.")
    parser.add_argument("prompt", help="Prompt text.")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--weights-file", default="model.npz")
    parser.add_argument("--weights-path", type=Path, default=None)
    parser.add_argument("--config-file", default="config.json")
    parser.add_argument("--safetensors-file", default="model.safetensors")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=1.0e-3)
    parser.add_argument("--fail-on-mismatch", action="store_true")
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


def run_tokenize(argv: list[str]) -> int:
    parser = build_tokenize_parser()
    args = parser.parse_args(argv)

    tokenizer_path = download_model_file(
        args.model_id,
        "tokenizer.json",
        endpoint=args.endpoint,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )
    tokenizer = HfTokenizer.from_file(tokenizer_path)
    token_ids = tokenizer.encode(args.text)
    print(f"tokenizer_file: {tokenizer_path}")
    print(f"vocab_size: {tokenizer.vocab_size}")
    print("ids: " + " ".join(str(token_id) for token_id in token_ids))
    print(f"decoded: {tokenizer.decode(token_ids)}")
    return 0


def run_inspect_weights(argv: list[str]) -> int:
    parser = build_inspect_weights_parser()
    args = parser.parse_args(argv)

    weights_path = download_model_file(
        args.model_id,
        args.filename,
        endpoint=args.endpoint,
        revision=args.revision,
        cache_dir=args.cache_dir,
        timeout=300.0,
    )
    manifest = load_torch_weight_manifest(weights_path)
    limit = None if args.limit == 0 else args.limit
    for line in manifest.summary_lines(limit=limit):
        print(line)
    return 0


def run_convert_weights(argv: list[str]) -> int:
    parser = build_convert_weights_parser()
    args = parser.parse_args(argv)

    weights_path = download_model_file(
        args.model_id,
        args.filename,
        endpoint=args.endpoint,
        revision=args.revision,
        cache_dir=args.cache_dir,
        timeout=300.0,
    )
    output_path = args.output
    if output_path is None:
        output_path = weights_path.with_suffix(".npz")

    manifest = convert_torch_weights_to_npz(
        weights_path,
        output_path,
        compressed=not args.uncompressed,
    )
    limit = None if args.limit == 0 else args.limit
    for line in manifest.summary_lines(limit=limit):
        print(line)
    return 0


def run_inspect_safetensors(argv: list[str]) -> int:
    parser = build_inspect_safetensors_parser()
    args = parser.parse_args(argv)

    weights_path = download_model_file(
        args.model_id,
        args.filename,
        endpoint=args.endpoint,
        revision=args.revision,
        cache_dir=args.cache_dir,
        timeout=300.0,
    )
    manifest = load_safetensors_weight_manifest(weights_path)
    limit = None if args.limit == 0 else args.limit
    for line in manifest.summary_lines(limit=limit):
        print(line)
    return 0


def run_convert_safetensors(argv: list[str]) -> int:
    parser = build_convert_safetensors_parser()
    args = parser.parse_args(argv)

    weights_path = download_model_file(
        args.model_id,
        args.filename,
        endpoint=args.endpoint,
        revision=args.revision,
        cache_dir=args.cache_dir,
        timeout=300.0,
    )
    output_path = args.output
    if output_path is None:
        output_path = weights_path.with_suffix(".npz")

    manifest = convert_safetensors_weights_to_npz(
        weights_path,
        output_path,
        compressed=not args.uncompressed,
    )
    limit = None if args.limit == 0 else args.limit
    for line in manifest.summary_lines(limit=limit):
        print(line)
    return 0


def run_probe_gpt_neo(argv: list[str]) -> int:
    parser = build_probe_gpt_neo_parser()
    args = parser.parse_args(argv)

    model, tokenizer, paths = load_numpy_gpt_neo_model(args)
    token_ids = tokenizer.encode(args.prompt)
    output = model.forward(
        ForwardInput(
            token_ids=token_ids,
            mode=ForwardMode.PREFILL,
        )
    )
    logits = output.logits
    print(f"config_file: {paths['config']}")
    print(f"weights_file: {paths['weights']}")
    print(f"prompt_token_ids: {' '.join(str(token_id) for token_id in token_ids)}")
    print(f"logits_size: {len(logits)}")
    print(f"cache_tokens: {len(output.cache.token_ids)}")
    top_id = max(range(len(logits)), key=lambda token_id: logits[token_id])
    print(f"top_token_id: {top_id}")
    print(f"top_token_text: {tokenizer.decode([top_id])!r}")
    return 0


def load_numpy_gpt_neo_model(args: argparse.Namespace) -> tuple[NumpyGptNeoLanguageModel, HfTokenizer, dict[str, Path]]:
    config_path = download_model_file(
        args.model_id,
        args.config_file,
        endpoint=args.endpoint,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )
    tokenizer_path = download_model_file(
        args.model_id,
        "tokenizer.json",
        endpoint=args.endpoint,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )
    weights_path = args.weights_path
    if weights_path is None:
        weights_path = model_cache_dir(
            args.model_id,
            revision=args.revision,
            cache_dir=args.cache_dir,
        ) / args.weights_file
    if not weights_path.exists():
        raise SystemExit(
            f"Missing NumPy weights file: {weights_path}\n"
            "Run `tiny-invoker convert-weights <model_id>` first."
        )

    tokenizer = HfTokenizer.from_file(tokenizer_path)
    model = NumpyGptNeoLanguageModel.from_files(
        config_path=config_path,
        weights_path=weights_path,
        tokenizer=tokenizer,
    )
    return model, tokenizer, {"config": config_path, "tokenizer": tokenizer_path, "weights": weights_path}


def run_probe_qwen2(argv: list[str]) -> int:
    parser = build_probe_qwen2_parser()
    args = parser.parse_args(argv)

    model, tokenizer, paths = load_numpy_qwen2_model(args)
    token_ids = tokenizer.encode(args.prompt)
    output = model.forward(
        ForwardInput(
            token_ids=token_ids,
            mode=ForwardMode.PREFILL,
        )
    )
    logits = output.logits
    print(f"config_file: {paths['config']}")
    print(f"weights_file: {paths['weights']}")
    print(f"prompt_token_ids: {' '.join(str(token_id) for token_id in token_ids)}")
    print(f"logits_size: {len(logits)}")
    print(f"cache_tokens: {len(output.cache.token_ids)}")
    top_id = max(range(len(logits)), key=lambda token_id: logits[token_id])
    print(f"top_token_id: {top_id}")
    print(f"top_token_text: {tokenizer.decode([top_id])!r}")
    return 0


def load_numpy_qwen2_model(args: argparse.Namespace) -> tuple[NumpyQwen2LanguageModel, HfTokenizer, dict[str, Path]]:
    config_path = download_model_file(
        args.model_id,
        args.config_file,
        endpoint=args.endpoint,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )
    tokenizer_path = download_model_file(
        args.model_id,
        "tokenizer.json",
        endpoint=args.endpoint,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )
    weights_path = args.weights_path
    if weights_path is None:
        weights_path = model_cache_dir(
            args.model_id,
            revision=args.revision,
            cache_dir=args.cache_dir,
        ) / args.weights_file
    if not weights_path.exists():
        raise SystemExit(
            f"Missing NumPy weights file: {weights_path}\n"
            "Run `tiny-invoker convert-safetensors <model_id>` first."
        )

    tokenizer = HfTokenizer.from_file(tokenizer_path)
    model = NumpyQwen2LanguageModel.from_files(
        config_path=config_path,
        weights_path=weights_path,
        tokenizer=tokenizer,
    )
    return model, tokenizer, {"config": config_path, "tokenizer": tokenizer_path, "weights": weights_path}


def run_generate_gpt_neo(argv: list[str]) -> int:
    parser = build_generate_gpt_neo_parser()
    args = parser.parse_args(argv)

    model, _, _ = load_numpy_gpt_neo_model(args)
    top_k = args.top_k if args.top_k > 0 else None
    engine = InferenceEngine(model=model)
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


def run_generate_qwen2(argv: list[str]) -> int:
    parser = build_generate_qwen2_parser()
    args = parser.parse_args(argv)

    model, _, _ = load_numpy_qwen2_model(args)
    top_k = args.top_k if args.top_k > 0 else None
    engine = InferenceEngine(model=model)
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


def run_serve_gpt_neo(argv: list[str]) -> int:
    parser = build_serve_gpt_neo_parser()
    args = parser.parse_args(argv)

    model, _, _ = load_numpy_gpt_neo_model(args)
    serve(
        engine=InferenceEngine(model=model),
        model_name=args.model_id,
        host=args.host,
        port=args.port,
    )
    return 0


def run_bench_gpt_neo(argv: list[str]) -> int:
    parser = build_bench_gpt_neo_parser()
    args = parser.parse_args(argv)
    return run_language_model_benchmark(
        args=args,
        benchmark_name="gpt_neo_runtime",
        load_model=load_numpy_gpt_neo_model,
    )


def run_bench_qwen2(argv: list[str]) -> int:
    parser = build_bench_qwen2_parser()
    args = parser.parse_args(argv)
    return run_language_model_benchmark(
        args=args,
        benchmark_name="qwen2_runtime",
        load_model=load_numpy_qwen2_model,
    )


def run_language_model_benchmark(
    args: argparse.Namespace,
    benchmark_name: str,
    load_model: ModelLoader,
) -> int:
    if args.max_new_tokens < 0:
        raise SystemExit("max_new_tokens must be non-negative.")
    if args.repeats <= 0:
        raise SystemExit("repeats must be positive.")
    if args.warmups < 0:
        raise SystemExit("warmups must be non-negative.")

    load_start = time.perf_counter()
    model, tokenizer, _ = load_model(args)
    load_ms = elapsed_ms(load_start)
    engine = InferenceEngine(model=model)
    top_k = args.top_k if args.top_k > 0 else None
    prompt_token_ids = tokenizer.encode(args.prompt)

    for _ in range(args.warmups):
        run_segmented_language_model_benchmark(
            model=model,
            tokenizer=tokenizer,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=top_k,
            seed=args.seed,
            profile=args.profile,
        )
        run_end_to_end_language_model_benchmark(
            engine=engine,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=top_k,
            seed=args.seed,
        )

    segmented_rows = [
        run_segmented_language_model_benchmark(
            model=model,
            tokenizer=tokenizer,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=top_k,
            seed=args.seed,
            profile=args.profile,
        )
        for _ in range(args.repeats)
    ]
    end_to_end_rows = [
        run_end_to_end_language_model_benchmark(
            engine=engine,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=top_k,
            seed=args.seed,
        )
        for _ in range(args.repeats)
    ]

    print(f"benchmark_name: {benchmark_name}")
    print(f"model_id: {args.model_id}")
    print(f"prompt: {args.prompt!r}")
    print(f"prompt_tokens: {len(prompt_token_ids)}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"temperature: {args.temperature}")
    print(f"top_k: {top_k}")
    print(f"repeats: {args.repeats}")
    print(f"warmups: {args.warmups}")
    print(f"profile: {args.profile}")
    print(f"model_load_ms_excluded: {load_ms:.2f}")
    for metric_name in SEGMENTED_BENCHMARK_METRIC_NAMES:
        print_benchmark_metric(metric_name, segmented_rows)
    if args.profile:
        for metric_name in PROFILE_METRIC_NAMES:
            print_benchmark_metric(f"profile_prefill_{metric_name}", segmented_rows)
        for metric_name in PROFILE_METRIC_NAMES:
            print_benchmark_metric(f"profile_decode_{metric_name}", segmented_rows)
    for metric_name in END_TO_END_BENCHMARK_METRIC_NAMES:
        print_benchmark_metric(metric_name, end_to_end_rows)
    if args.sample_chars > 0 and end_to_end_rows:
        print("sample_output_prefix:")
        print(end_to_end_rows[0]["text"][: args.sample_chars])
    if args.json:
        payload = build_benchmark_json_payload(
            args=args,
            benchmark_name=benchmark_name,
            top_k=top_k,
            prompt_token_count=len(prompt_token_ids),
            load_ms=load_ms,
            segmented_rows=segmented_rows,
            end_to_end_rows=end_to_end_rows,
        )
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


def run_compare_gpt_neo(argv: list[str]) -> int:
    parser = build_compare_gpt_neo_parser()
    args = parser.parse_args(argv)
    if args.top_k <= 0:
        raise SystemExit("top_k must be positive.")
    if args.tolerance < 0:
        raise SystemExit("tolerance must be non-negative.")

    model, tokenizer, paths = load_numpy_gpt_neo_model(args)
    token_ids = tokenizer.encode(args.prompt)
    context_token_ids = token_ids[:] or [tokenizer.bos_id]
    numpy_output = model.forward(
        ForwardInput(
            token_ids=context_token_ids,
            mode=ForwardMode.PREFILL,
        )
    )

    reference_logits = load_hf_reference_gpt_neo_logits(args, context_token_ids)
    np = require_numpy()
    numpy_logits = np.asarray(numpy_output.logits, dtype=np.float32)
    hf_logits = np.asarray(reference_logits, dtype=np.float32)
    if numpy_logits.shape != hf_logits.shape:
        raise SystemExit(f"logits shape mismatch: numpy={numpy_logits.shape}, hf={hf_logits.shape}.")

    diff = np.abs(numpy_logits - hf_logits)
    max_abs_diff = float(np.max(diff))
    mean_abs_diff = float(np.mean(diff))
    numpy_top_ids = top_token_ids(numpy_logits, args.top_k)
    hf_top_ids = top_token_ids(hf_logits, args.top_k)
    top1_match = bool(numpy_top_ids and hf_top_ids and numpy_top_ids[0] == hf_top_ids[0])
    top_k_overlap = len(set(numpy_top_ids) & set(hf_top_ids))
    within_tolerance = max_abs_diff <= args.tolerance

    print("comparison_name: gpt_neo_hf_logits")
    print(f"model_id: {args.model_id}")
    print(f"prompt: {args.prompt!r}")
    print(f"prompt_token_ids: {' '.join(str(token_id) for token_id in context_token_ids)}")
    print(f"config_file: {paths['config']}")
    print(f"numpy_weights_file: {paths['weights']}")
    print(f"torch_weights_file: {model_cache_dir(args.model_id, revision=args.revision, cache_dir=args.cache_dir) / args.torch_weights_file}")
    print(f"logits_size: {numpy_logits.shape[0]}")
    print(f"max_abs_diff: {max_abs_diff:.8f}")
    print(f"mean_abs_diff: {mean_abs_diff:.8f}")
    print(f"tolerance: {args.tolerance:.8f}")
    print(f"within_tolerance: {within_tolerance}")
    print(f"top1_match: {top1_match}")
    print(f"top_{args.top_k}_overlap: {top_k_overlap}/{args.top_k}")
    print("top_tokens:")
    print("rank numpy_id numpy_text numpy_logit hf_logit_at_numpy_id abs_diff hf_rank_id hf_rank_text")
    for rank, (numpy_id, hf_id) in enumerate(zip(numpy_top_ids, hf_top_ids), start=1):
        numpy_logit = float(numpy_logits[numpy_id])
        hf_logit_at_numpy_id = float(hf_logits[numpy_id])
        token_diff = abs(numpy_logit - hf_logit_at_numpy_id)
        print(
            f"{rank} "
            f"{numpy_id} {tokenizer.decode([numpy_id])!r} "
            f"{numpy_logit:.6f} {hf_logit_at_numpy_id:.6f} {token_diff:.6f} "
            f"{hf_id} {tokenizer.decode([hf_id])!r}"
        )

    if args.fail_on_mismatch and not within_tolerance:
        return 1
    return 0


def run_compare_qwen2(argv: list[str]) -> int:
    parser = build_compare_qwen2_parser()
    args = parser.parse_args(argv)
    if args.top_k <= 0:
        raise SystemExit("top_k must be positive.")
    if args.tolerance < 0:
        raise SystemExit("tolerance must be non-negative.")

    model, tokenizer, paths = load_numpy_qwen2_model(args)
    token_ids = tokenizer.encode(args.prompt)
    context_token_ids = token_ids[:] or [tokenizer.bos_id]
    numpy_output = model.forward(
        ForwardInput(
            token_ids=context_token_ids,
            mode=ForwardMode.PREFILL,
        )
    )

    reference_logits = load_hf_reference_qwen2_logits(args, context_token_ids)
    np = require_numpy()
    numpy_logits = np.asarray(numpy_output.logits, dtype=np.float32)
    hf_logits = np.asarray(reference_logits, dtype=np.float32)
    if numpy_logits.shape != hf_logits.shape:
        raise SystemExit(f"logits shape mismatch: numpy={numpy_logits.shape}, hf={hf_logits.shape}.")

    diff = np.abs(numpy_logits - hf_logits)
    max_abs_diff = float(np.max(diff))
    mean_abs_diff = float(np.mean(diff))
    numpy_top_ids = top_token_ids(numpy_logits, args.top_k)
    hf_top_ids = top_token_ids(hf_logits, args.top_k)
    top1_match = bool(numpy_top_ids and hf_top_ids and numpy_top_ids[0] == hf_top_ids[0])
    top_k_overlap = len(set(numpy_top_ids) & set(hf_top_ids))
    within_tolerance = max_abs_diff <= args.tolerance

    print("comparison_name: qwen2_hf_logits")
    print(f"model_id: {args.model_id}")
    print(f"prompt: {args.prompt!r}")
    print(f"prompt_token_ids: {' '.join(str(token_id) for token_id in context_token_ids)}")
    print(f"config_file: {paths['config']}")
    print(f"numpy_weights_file: {paths['weights']}")
    print(f"safetensors_file: {model_cache_dir(args.model_id, revision=args.revision, cache_dir=args.cache_dir) / args.safetensors_file}")
    print(f"logits_size: {numpy_logits.shape[0]}")
    print(f"max_abs_diff: {max_abs_diff:.8f}")
    print(f"mean_abs_diff: {mean_abs_diff:.8f}")
    print(f"tolerance: {args.tolerance:.8f}")
    print(f"within_tolerance: {within_tolerance}")
    print(f"top1_match: {top1_match}")
    print(f"top_{args.top_k}_overlap: {top_k_overlap}/{args.top_k}")
    print("top_tokens:")
    print("rank numpy_id numpy_text numpy_logit hf_logit_at_numpy_id abs_diff hf_rank_id hf_rank_text")
    for rank, (numpy_id, hf_id) in enumerate(zip(numpy_top_ids, hf_top_ids), start=1):
        numpy_logit = float(numpy_logits[numpy_id])
        hf_logit_at_numpy_id = float(hf_logits[numpy_id])
        token_diff = abs(numpy_logit - hf_logit_at_numpy_id)
        print(
            f"{rank} "
            f"{numpy_id} {tokenizer.decode([numpy_id])!r} "
            f"{numpy_logit:.6f} {hf_logit_at_numpy_id:.6f} {token_diff:.6f} "
            f"{hf_id} {tokenizer.decode([hf_id])!r}"
        )

    if args.fail_on_mismatch and not within_tolerance:
        return 1
    return 0


def load_hf_reference_gpt_neo_logits(args: argparse.Namespace, token_ids: list[int]) -> Any:
    try:
        import torch
        from transformers import AutoModelForCausalLM
        from transformers.utils import logging as transformers_logging
    except ImportError as error:
        raise RuntimeError(
            "HF comparison requires optional compare dependencies. Install them with "
            "`python3 -m pip install '.[compare]'` from this repository."
        ) from error

    download_model_file(
        args.model_id,
        args.config_file,
        endpoint=args.endpoint,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )
    download_model_file(
        args.model_id,
        args.torch_weights_file,
        endpoint=args.endpoint,
        revision=args.revision,
        cache_dir=args.cache_dir,
        timeout=300.0,
    )
    model_dir = model_cache_dir(args.model_id, revision=args.revision, cache_dir=args.cache_dir)
    transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()
    reference_model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
    )
    reference_model.eval()
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    with torch.no_grad():
        output = reference_model(input_ids=input_ids)
    return output.logits[0, -1].detach().cpu().numpy()


def load_hf_reference_qwen2_logits(args: argparse.Namespace, token_ids: list[int]) -> Any:
    try:
        import torch
        from transformers import AutoModelForCausalLM
        from transformers.utils import logging as transformers_logging
    except ImportError as error:
        raise RuntimeError(
            "HF comparison requires optional compare dependencies. Install them with "
            "`python3 -m pip install '.[compare]'` from this repository."
        ) from error

    download_model_file(
        args.model_id,
        args.config_file,
        endpoint=args.endpoint,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )
    download_model_file(
        args.model_id,
        args.safetensors_file,
        endpoint=args.endpoint,
        revision=args.revision,
        cache_dir=args.cache_dir,
        timeout=300.0,
    )
    model_dir = model_cache_dir(args.model_id, revision=args.revision, cache_dir=args.cache_dir)
    transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()
    reference_model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
    )
    reference_model.eval()
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    with torch.no_grad():
        output = reference_model(input_ids=input_ids)
    return output.logits[0, -1].detach().cpu().numpy()


def top_token_ids(logits: Any, top_k: int) -> list[int]:
    np = require_numpy()
    limit = min(top_k, int(logits.shape[0]))
    return [int(token_id) for token_id in np.argsort(logits)[-limit:][::-1]]


def elapsed_ms(start_time: float) -> float:
    return (time.perf_counter() - start_time) * 1000.0


def run_segmented_language_model_benchmark(
    model: Any,
    tokenizer: HfTokenizer,
    prompt_token_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    seed: int | None,
    profile: bool = False,
) -> BenchmarkRow:
    rng = random.Random(seed)
    prefill_start = time.perf_counter()
    output, prefill_profile = run_profiled_forward(
        model,
        ForwardInput(
            token_ids=prompt_token_ids,
            mode=ForwardMode.PREFILL,
        ),
        profile=profile,
    )
    prefill_ms = elapsed_ms(prefill_start)

    logits = output.logits
    cache = output.cache
    sampler_ms: list[float] = []
    decode_ms: list[float] = []
    decode_profiles: list[dict[str, float]] = []
    generated_token_ids: list[int] = []
    for step in range(max_new_tokens):
        sample_start = time.perf_counter()
        token_id = choose_token(
            logits,
            rng=rng,
            temperature=temperature,
            top_k=top_k,
            blocked_token_ids=tokenizer.special_token_ids,
        )
        sampler_ms.append(elapsed_ms(sample_start))
        generated_token_ids.append(token_id)

        if step < max_new_tokens - 1:
            decode_start = time.perf_counter()
            output, decode_profile = run_profiled_forward(
                model,
                ForwardInput(
                    token_ids=[token_id],
                    mode=ForwardMode.DECODE,
                    cache=cache,
                ),
                profile=profile,
            )
            decode_ms.append(elapsed_ms(decode_start))
            if profile:
                decode_profiles.append(decode_profile)
            logits = output.logits
            cache = output.cache

    sampler_total_ms = sum(sampler_ms)
    decode_forward_total_ms = sum(decode_ms)
    subsequent_token_ms = [
        decode_ms[index] + sampler_ms[index + 1]
        for index in range(min(len(decode_ms), max(0, len(sampler_ms) - 1)))
    ]
    subsequent_token_total_ms = sum(subsequent_token_ms)
    prefill_tokens_per_second = (
        len(prompt_token_ids) / (prefill_ms / 1000.0) if prefill_ms > 0 else 0.0
    )
    ttft_ms = prefill_ms + sampler_ms[0] if sampler_ms else 0.0
    tpot_ms = mean(subsequent_token_ms) if subsequent_token_ms else 0.0
    decode_tokens_per_second = (
        len(subsequent_token_ms) / (subsequent_token_total_ms / 1000.0)
        if subsequent_token_total_ms > 0
        else 0.0
    )
    model_decode_tokens_per_second = (
        len(decode_ms) / (decode_forward_total_ms / 1000.0)
        if decode_forward_total_ms > 0
        else 0.0
    )

    row: BenchmarkRow = {
        "prompt_tokens": len(prompt_token_ids),
        "generated_tokens": max_new_tokens,
        "decode_steps": len(decode_ms),
        "prefill_ms": prefill_ms,
        "prefill_tokens_per_second": prefill_tokens_per_second,
        "ttft_ms": ttft_ms,
        "decode_forward_ms": mean(decode_ms) if decode_ms else 0.0,
        "decode_forward_total_ms": decode_forward_total_ms,
        "sampler_with_mask_ms": mean(sampler_ms) if sampler_ms else 0.0,
        "sampler_total_ms": sampler_total_ms,
        "tpot_ms": tpot_ms,
        "decode_tokens_per_second": decode_tokens_per_second,
        "model_decode_tokens_per_second": model_decode_tokens_per_second,
        "text": tokenizer.decode(generated_token_ids),
    }
    if profile:
        for metric_name in PROFILE_METRIC_NAMES:
            row[f"profile_prefill_{metric_name}"] = prefill_profile.get(metric_name, 0.0)
            decode_values = [profile_row.get(metric_name, 0.0) for profile_row in decode_profiles]
            row[f"profile_decode_{metric_name}"] = mean(decode_values) if decode_values else 0.0
    return row


def run_profiled_forward(
    model: Any,
    request: ForwardInput,
    profile: bool,
) -> tuple[Any, dict[str, float]]:
    if profile:
        return model.profile_forward(request)
    return model.forward(request), {}


def run_end_to_end_language_model_benchmark(
    engine: InferenceEngine,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    seed: int | None,
) -> BenchmarkRow:
    start = time.perf_counter()
    result = engine.generate(
        prompt,
        config=GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=seed,
            trace=False,
        ),
    )
    total_ms = elapsed_ms(start)
    tokens_per_second = max_new_tokens / (total_ms / 1000.0) if total_ms > 0 else 0.0
    return {
        "end_to_end_ms": total_ms,
        "end_to_end_tokens_per_second": tokens_per_second,
        "text": result.text,
    }


def benchmark_metric_stats(name: str, rows: list[BenchmarkRow]) -> dict[str, float]:
    values = [float(row[name]) for row in rows]
    value_mean = mean(values) if values else 0.0
    value_stdev = stdev(values) if len(values) > 1 else 0.0
    return {
        "avg": value_mean,
        "stdev": value_stdev,
    }


def print_benchmark_metric(name: str, rows: list[BenchmarkRow]) -> None:
    stats = benchmark_metric_stats(name, rows)
    print(f"{name}_avg: {stats['avg']:.4f}")
    print(f"{name}_stdev: {stats['stdev']:.4f}")


def build_benchmark_json_payload(
    args: argparse.Namespace,
    benchmark_name: str,
    top_k: int | None,
    prompt_token_count: int,
    load_ms: float,
    segmented_rows: list[BenchmarkRow],
    end_to_end_rows: list[BenchmarkRow],
) -> dict[str, object]:
    metric_names = list(SEGMENTED_BENCHMARK_METRIC_NAMES) + list(
        END_TO_END_BENCHMARK_METRIC_NAMES
    )
    if args.profile:
        metric_names.extend(f"profile_prefill_{name}" for name in PROFILE_METRIC_NAMES)
        metric_names.extend(f"profile_decode_{name}" for name in PROFILE_METRIC_NAMES)

    metrics: dict[str, dict[str, float]] = {}
    for metric_name in metric_names:
        rows = (
            end_to_end_rows
            if metric_name in END_TO_END_BENCHMARK_METRIC_NAMES
            else segmented_rows
        )
        metrics[metric_name] = benchmark_metric_stats(metric_name, rows)

    return {
        "benchmark_name": benchmark_name,
        "model_id": args.model_id,
        "prompt": args.prompt,
        "prompt_tokens": prompt_token_count,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": top_k,
        "repeats": args.repeats,
        "warmups": args.warmups,
        "profile": args.profile,
        "model_load_ms_excluded": load_ms,
        "metrics": metrics,
    }


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "inspect-model":
        return run_inspect_model(args[1:])
    if args and args[0] == "tokenize":
        return run_tokenize(args[1:])
    if args and args[0] == "inspect-weights":
        return run_inspect_weights(args[1:])
    if args and args[0] == "convert-weights":
        return run_convert_weights(args[1:])
    if args and args[0] == "inspect-safetensors":
        return run_inspect_safetensors(args[1:])
    if args and args[0] == "convert-safetensors":
        return run_convert_safetensors(args[1:])
    if args and args[0] == "probe-gpt-neo":
        return run_probe_gpt_neo(args[1:])
    if args and args[0] == "probe-qwen2":
        return run_probe_qwen2(args[1:])
    if args and args[0] == "generate-gpt-neo":
        return run_generate_gpt_neo(args[1:])
    if args and args[0] == "generate-qwen2":
        return run_generate_qwen2(args[1:])
    if args and args[0] == "serve-gpt-neo":
        return run_serve_gpt_neo(args[1:])
    if args and args[0] == "bench-gpt-neo":
        return run_bench_gpt_neo(args[1:])
    if args and args[0] == "bench-qwen2":
        return run_bench_qwen2(args[1:])
    if args and args[0] == "compare-gpt-neo":
        return run_compare_gpt_neo(args[1:])
    if args and args[0] == "compare-qwen2":
        return run_compare_qwen2(args[1:])
    return run_generate(args)

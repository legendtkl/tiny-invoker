# tiny-invoker

`tiny-invoker` is a learning-oriented model inference engine. The first version is intentionally small: it uses a character-level bigram language model so the whole inference path is easy to inspect.

## Design Constraints

- Runs on macOS with the system `python3`.
- Keeps the core demo simple; real Hugging Face tokenizers use the small `tokenizers` dependency.
- Uses optional `numpy` and `torch` only for weight inspection and later weight conversion.
- Keeps the repository small enough to read file by file.
- Optimizes for learning the inference loop before adding fast tensor libraries.

## What This Version Teaches

The core loop of most text-generation inference engines is:

1. Tokenize input text into token ids.
2. Run prefill over the prompt to prepare reusable state.
3. Ask the model for scores for the first generated token.
4. Decode one token at a time, reusing the cache.
5. Convert scores into probabilities.
6. Sample or choose the next token.
7. Append the new token and repeat.
8. Decode token ids back into text.

The demo model keeps that loop easy to inspect. Hugging Face tokenizer and
weight-inspection commands use focused dependencies instead of a full inference
runtime.

## Docs

- [Transformer runtime walkthrough](docs/transformer.md): visual walkthrough of
  `DecoderOnlyTransformer.forward()`, Transformer blocks, self-attention, K/V
  cache, prefill, and decode. Includes Chinese and English explanations.

## Prefill And Decode

The engine separates generation into two phases:

- `prefill`: process the prompt once and return logits plus a cache.
- `decode_one`: process one newly generated token with the existing cache.

The model interface is still a single `forward` method. The engine calls it with
`ForwardMode.PREFILL` or `ForwardMode.DECODE`, which mirrors production systems
where scheduling stages are separate from model architecture code.

In a real Transformer, the cache stores attention K/V tensors for each layer.
The current bigram demo model stores token history instead, so the same engine
shape stays easy to read before we add real attention.

## Run

From this repository:

```bash
PYTHONPATH=src python3 -m tiny_invoker "tiny" --max-new-tokens 80 --seed 7
```

Inspect the target TinyStories model repository:

```bash
PYTHONPATH=src python3 -m tiny_invoker inspect-model roneneldan/TinyStories-33M
```

Tokenize text with the real TinyStories tokenizer:

```bash
PYTHONPATH=src python3 -m tiny_invoker tokenize roneneldan/TinyStories-33M "Once upon a time"
```

Inspect PyTorch weight names and shapes:

```bash
python3 -m pip install '.[weights]'
PYTHONPATH=src python3 -m tiny_invoker inspect-weights roneneldan/TinyStories-33M --limit 40
```

Convert PyTorch weights to a NumPy file for the later runtime path:

```bash
PYTHONPATH=src python3 -m tiny_invoker convert-weights roneneldan/TinyStories-33M
```

Convert a single-file safetensors checkpoint to NumPy:

```bash
python3 -m pip install '.[weights]'
PYTHONPATH=src python3 -m tiny_invoker convert-safetensors Qwen/Qwen2.5-0.5B --filename model.safetensors
```

This currently expects one safetensors file. Sharded checkpoints with
`model.safetensors.index.json` need a later shard-merging step.

Probe the NumPy GPT-Neo runtime skeleton:

```bash
python3 -m pip install '.[runtime]'
PYTHONPATH=src python3 -m tiny_invoker probe-gpt-neo roneneldan/TinyStories-33M "Once upon a time"
```

Generate with the NumPy GPT-Neo runtime:

```bash
PYTHONPATH=src python3 -m tiny_invoker generate-gpt-neo roneneldan/TinyStories-33M "Once upon a time" --max-new-tokens 12 --temperature 0
```

Benchmark the NumPy GPT-Neo runtime:

```bash
PYTHONPATH=src python3 -m tiny_invoker bench-gpt-neo roneneldan/TinyStories-33M "Once upon a time" --max-new-tokens 128 --temperature 0 --top-k 20 --profile --json --json-output benchmarks/baseline/gpt-neo.jsonl
```

This is the baseline command for optimization work. It prints:

- `prefill_ms` and `prefill_tokens_per_second`: prompt processing cost.
- `ttft_ms`: approximate time to first generated token.
- `tpot_ms` and `decode_tokens_per_second`: per-token decode cost after the
  first generated token.
- `model_decode_tokens_per_second`: decode forward-only throughput, excluding
  sampler time.
- `end_to_end_ms` and `end_to_end_tokens_per_second`: full engine wall-clock
  generation cost.

With `--profile`, the benchmark also prints internal prefill and per-token
decode timing for embedding, Transformer blocks, attention internals, MLP
internals, final norm, and LM head. With `--json`, it prints a final
machine-readable JSON line. With `--json-output`, it appends the same payload to
a JSONL file so baseline and optimized runs can be compared later. Each metric
records `avg`, `stdev`, `p50`, and `p95`; prefer multiple `--repeats` when
studying optimization effects on macOS CPU.

Compare two benchmark JSONL files:

```bash
PYTHONPATH=src python3 -m tiny_invoker compare-bench benchmarks/baseline/qwen2.jsonl benchmarks/optimized/qwen2.jsonl
PYTHONPATH=src python3 -m tiny_invoker compare-bench benchmarks/baseline/qwen2.jsonl benchmarks/optimized/qwen2.jsonl --all-records --stat p50
```

Compare NumPy GPT-Neo logits with Hugging Face Transformers:

```bash
python3 -m pip install '.[compare]'
PYTHONPATH=src python3 -m tiny_invoker compare-gpt-neo roneneldan/TinyStories-33M "Once upon a time" --top-k 10
```

This prints max/mean absolute logit error, top-1 agreement, top-k overlap, and
a token-level top-k table. Use `--fail-on-mismatch` when you want the command to
return a non-zero exit code if the max logit error exceeds `--tolerance`.

Probe, generate, or compare with the NumPy Qwen2 runtime after converting the
safetensors weights:

```bash
PYTHONPATH=src python3 -m tiny_invoker probe-qwen2 Qwen/Qwen2.5-0.5B "Hello"
PYTHONPATH=src python3 -m tiny_invoker generate-qwen2 Qwen/Qwen2.5-0.5B "Hello" --max-new-tokens 8 --temperature 0
PYTHONPATH=src python3 -m tiny_invoker bench-qwen2 Qwen/Qwen2.5-0.5B "Hello" --max-new-tokens 8 --temperature 0 --profile --json --json-output benchmarks/baseline/qwen2.jsonl
PYTHONPATH=src python3 -m tiny_invoker compare-qwen2 Qwen/Qwen2.5-0.5B "Hello" --top-k 10
```

`bench-qwen2` uses the same baseline metrics as `bench-gpt-neo`, but runs
through the Qwen2 adapter with RoPE, RMSNorm, SwiGLU MLP, and grouped-query
attention.

Serve the NumPy GPT-Neo runtime locally:

```bash
PYTHONPATH=src python3 -m tiny_invoker serve-gpt-neo roneneldan/TinyStories-33M --host 127.0.0.1 --port 8000
curl -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Once upon a time","max_new_tokens":8,"temperature":0,"top_k":20}'
```

Benchmark a running server from the client side:

```bash
PYTHONPATH=src python3 -m tiny_invoker bench-server --url http://127.0.0.1:8000/generate --prompt "Once upon a time" --max-new-tokens 8 --requests 8 --concurrency 2 --json --json-output benchmarks/baseline/server.jsonl
```

The current server response is not streaming, so `bench-server` reports full
request latency, time to first byte, non-streaming token time, request
throughput, and generated-token throughput. True TTFT/ITL serving metrics will
become meaningful after a streaming endpoint is added.

The GPT-Neo runtime now uses a small decoder-only Transformer backbone. GPT-Neo
specific code maps Hugging Face config and weight names into that backbone, while
the shared Transformer code runs embedding, attention, MLP, residual paths, final
norm, LM head, and per-layer K/V cache for a single request on CPU. The shared
backbone also has the Qwen2-style pieces needed by modern decoder models:
RMSNorm, RoPE, SwiGLU, and grouped-query attention. It is built for learning,
not vLLM/SGLang-style batching or optimized serving yet. The hot path caches
transposed linear weights and exposes optional profiling so decode performance
work has a repeatable baseline.

To see each generation step:

```bash
PYTHONPATH=src python3 -m tiny_invoker "学习" --max-new-tokens 12 --seed 3 --trace
```

## Test

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

## Project Map

- `src/tiny_invoker/tokenizer.py`: turns text into token ids and back.
- `src/tiny_invoker/hf.py`: inspects Hugging Face model metadata and caches model files.
- `src/tiny_invoker/weights.py`: inspects PyTorch weight names, shapes, and dtypes.
- `src/tiny_invoker/transformer.py`: shared decoder-only Transformer execution code.
- `src/tiny_invoker/gpt_neo.py`: adapts GPT-Neo config and NumPy weights into the shared Transformer.
- `src/tiny_invoker/qwen2.py`: adapts Qwen2 config and NumPy weights into the shared Transformer.
- `src/tiny_invoker/interfaces.py`: defines the minimum model interface.
- `src/tiny_invoker/model.py`: a tiny bigram language model.
- `src/tiny_invoker/sampler.py`: softmax, top-k filtering, and token sampling.
- `src/tiny_invoker/engine.py`: the inference loop and generation config.
- `src/tiny_invoker/server.py`: a minimal local HTTP serving layer.
- `src/tiny_invoker/demo.py`: builds a demo engine from a small built-in corpus.
- `src/tiny_invoker/cli.py`: command line entry point.

## Next Learning Steps

Good next steps after the current GPT-Neo and Qwen2 runtime:

1. Add safetensors index/shard merging for larger Qwen checkpoints.
2. Add Qwen3 dense support, including QK-Norm.
3. Add streaming HTTP responses and an OpenAI-compatible local endpoint.
4. Add simple request batching, then continuous batching.
5. Replace the contiguous K/V cache with a small page/block-based cache manager.

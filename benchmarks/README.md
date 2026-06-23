# Benchmark Records

This directory is for local JSONL benchmark outputs.

Recommended baseline layout:

```text
benchmarks/
  baseline/
    gpt-neo.jsonl
    qwen2.jsonl
    server.jsonl
  optimized/
    qwen2-rope-cache.jsonl
    qwen2-kv-cache.jsonl
```

Recommended Qwen2 baseline prompts:

```bash
PYTHONPATH=src python3 -m tiny_invoker bench-qwen2 Qwen/Qwen2.5-0.5B "Hello" --max-new-tokens 8 --temperature 0 --profile --json-output benchmarks/baseline/qwen2.jsonl
PYTHONPATH=src python3 -m tiny_invoker bench-qwen2 Qwen/Qwen2.5-0.5B "Explain KV cache in one paragraph." --max-new-tokens 32 --temperature 0 --profile --json-output benchmarks/baseline/qwen2.jsonl
PYTHONPATH=src python3 -m tiny_invoker bench-qwen2 Qwen/Qwen2.5-0.5B "Write a short story about a robot learning to read." --max-new-tokens 64 --temperature 0 --profile --json-output benchmarks/baseline/qwen2.jsonl
```

Compare a baseline with a later optimized run:

```bash
PYTHONPATH=src python3 -m tiny_invoker compare-bench benchmarks/baseline/qwen2.jsonl benchmarks/optimized/qwen2-rope-cache.jsonl
```

JSONL files are machine-specific measurements. Keep the commands stable, and
compare runs from the same machine when studying optimization effects.

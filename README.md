# tiny-invoker

`tiny-invoker` is a learning-oriented model inference engine. The first version is intentionally small: it uses a character-level bigram language model so the whole inference path is easy to inspect.

## Design Constraints

- Runs on macOS with the system `python3`.
- Uses only the Python standard library for now.
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

This project implements that loop in plain Python, without external dependencies.

## Prefill And Decode

The engine separates generation into two phases:

- `prefill`: process the prompt once and return logits plus a cache.
- `decode_one`: process one newly generated token with the existing cache.

In a real Transformer, the cache stores attention K/V tensors for each layer.
The current bigram demo model stores token history instead, so the same engine
shape stays easy to read before we add real attention.

## Run

From this repository:

```bash
PYTHONPATH=src python3 -m tiny_invoker "tiny" --max-new-tokens 80 --seed 7
```

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
- `src/tiny_invoker/interfaces.py`: defines the minimum model interface.
- `src/tiny_invoker/model.py`: a tiny bigram language model.
- `src/tiny_invoker/sampler.py`: softmax, top-k filtering, and token sampling.
- `src/tiny_invoker/engine.py`: the inference loop and generation config.
- `src/tiny_invoker/demo.py`: builds a demo engine from a small built-in corpus.
- `src/tiny_invoker/cli.py`: command line entry point.

## Next Learning Steps

Good next steps after this first runnable version:

1. Replace the bigram model with a tiny neural network layer.
2. Add tensor operations with NumPy.
3. Add a Transformer block.
4. Load weights from a small checkpoint file.
5. Add batching and key-value cache concepts.

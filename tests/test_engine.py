from dataclasses import dataclass
import unittest

from tiny_invoker.demo import build_demo_engine
from tiny_invoker.engine import GenerationConfig, InferenceEngine
from tiny_invoker.interfaces import ForwardInput, ForwardMode, ForwardOutput
from tiny_invoker.tokenizer import CharTokenizer


@dataclass(frozen=True)
class CountingCache:
    token_ids: list[int]


class CountingModel:
    def __init__(self) -> None:
        self.tokenizer = CharTokenizer.from_text("ab")
        self.prefill_calls = 0
        self.decode_calls = 0
        self.prefill_token_ids: list[int] | None = None

    def _logits_for(self, token: str) -> list[float]:
        logits = [0.0 for _ in range(self.tokenizer.vocab_size)]
        logits[self.tokenizer.token_to_id[token]] = 10.0
        return logits

    def forward(self, request: ForwardInput) -> ForwardOutput:
        if request.mode == ForwardMode.PREFILL:
            self.prefill_calls += 1
            self.prefill_token_ids = request.token_ids[:]
            return ForwardOutput(
                logits=self._logits_for("a"),
                cache=CountingCache(token_ids=request.token_ids[:]),
            )

        if request.mode == ForwardMode.DECODE:
            if not isinstance(request.cache, CountingCache):
                raise TypeError("CountingModel expected CountingCache.")
            if len(request.token_ids) != 1:
                raise ValueError("CountingModel decode expects exactly one token id.")

            self.decode_calls += 1
            return ForwardOutput(
                logits=self._logits_for("b"),
                cache=CountingCache(token_ids=request.cache.token_ids + request.token_ids),
            )

        raise ValueError(f"Unsupported mode: {request.mode}.")


class InferenceEngineTest(unittest.TestCase):
    def test_generate_appends_requested_number_of_tokens(self) -> None:
        engine = build_demo_engine()

        result = engine.generate("tiny", config=GenerationConfig(max_new_tokens=5, seed=1))

        self.assertTrue(result.text.startswith("tiny"))
        self.assertEqual(len(result.token_ids), len(engine.model.tokenizer.encode("tiny")) + 5)

    def test_generate_is_reproducible_with_seed(self) -> None:
        engine = build_demo_engine()

        first = engine.generate("tiny", config=GenerationConfig(max_new_tokens=20, seed=3))
        second = engine.generate("tiny", config=GenerationConfig(max_new_tokens=20, seed=3))

        self.assertEqual(first.text, second.text)

    def test_trace_records_each_step(self) -> None:
        engine = build_demo_engine()

        result = engine.generate(
            "学习",
            config=GenerationConfig(max_new_tokens=3, seed=2, trace=True),
        )

        self.assertEqual(len(result.steps), 3)
        self.assertTrue(result.steps[0].candidates)

    def test_generation_config_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            GenerationConfig(max_new_tokens=-1).validate()
        with self.assertRaises(ValueError):
            GenerationConfig(temperature=-0.1).validate()
        with self.assertRaises(ValueError):
            GenerationConfig(top_k=0).validate()

    def test_engine_runs_prefill_mode_then_decode_mode(self) -> None:
        model = CountingModel()
        engine = InferenceEngine(model=model)

        result = engine.generate(
            "",
            config=GenerationConfig(max_new_tokens=2, temperature=0),
        )

        self.assertEqual(result.text, "ab")
        self.assertEqual(model.prefill_calls, 1)
        self.assertEqual(model.decode_calls, 1)
        self.assertEqual(model.prefill_token_ids, [model.tokenizer.bos_id])


if __name__ == "__main__":
    unittest.main()

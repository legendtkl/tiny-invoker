import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tiny_invoker.gpt_neo import NumpyGptNeoConfig


def require_numpy_for_test():
    try:
        import numpy as np
    except ImportError:
        raise unittest.SkipTest("NumPy is only required for optional GPT-Neo runtime tests.")
    return np


class NumpyGptNeoConfigTest(unittest.TestCase):
    def test_loads_gpt_neo_config(self) -> None:
        config = NumpyGptNeoConfig.from_dict(
            {
                "model_type": "gpt_neo",
                "vocab_size": 4,
                "hidden_size": 3,
                "max_position_embeddings": 8,
                "layer_norm_epsilon": 1e-5,
                "num_layers": 1,
                "num_heads": 1,
            }
        )

        self.assertEqual(config.vocab_size, 4)
        self.assertEqual(config.hidden_size, 3)

    def test_rejects_non_gpt_neo_config(self) -> None:
        with self.assertRaises(ValueError):
            NumpyGptNeoConfig.from_dict({"model_type": "llama"})


class NumpyGptNeoLanguageModelTest(unittest.TestCase):
    def test_runtime_skeleton_prefill_and_decode(self) -> None:
        np = require_numpy_for_test()

        from tiny_invoker.gpt_neo import NumpyGptNeoLanguageModel
        from tiny_invoker.interfaces import ForwardInput, ForwardMode
        from tiny_invoker.tokenizer import CharTokenizer

        tokenizer = CharTokenizer.from_text("ab")
        with TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            weights_path = Path(tmp_dir) / "weights.npz"
            config_path.write_text(
                json.dumps(
                    {
                        "model_type": "gpt_neo",
                        "vocab_size": tokenizer.vocab_size,
                        "hidden_size": 3,
                        "max_position_embeddings": 8,
                        "layer_norm_epsilon": 1e-5,
                        "num_layers": 1,
                        "num_heads": 1,
                    }
                ),
                encoding="utf-8",
            )
            np.savez(
                weights_path,
                **{
                    "transformer.wte.weight": np.arange(tokenizer.vocab_size * 3, dtype=np.float32).reshape(
                        tokenizer.vocab_size,
                        3,
                    ),
                    "transformer.wpe.weight": np.zeros((8, 3), dtype=np.float32),
                    "transformer.ln_f.weight": np.ones((3,), dtype=np.float32),
                    "transformer.ln_f.bias": np.zeros((3,), dtype=np.float32),
                },
            )

            model = NumpyGptNeoLanguageModel.from_files(
                config_path=config_path,
                weights_path=weights_path,
                tokenizer=tokenizer,
            )
            prefill = model.forward(
                ForwardInput(
                    token_ids=tokenizer.encode("a"),
                    mode=ForwardMode.PREFILL,
                )
            )
            decode = model.forward(
                ForwardInput(
                    token_ids=tokenizer.encode("b"),
                    mode=ForwardMode.DECODE,
                    cache=prefill.cache,
                )
            )

        self.assertEqual(len(prefill.logits), tokenizer.vocab_size)
        self.assertEqual(prefill.cache.token_ids, tokenizer.encode("a"))
        self.assertEqual(decode.cache.token_ids, tokenizer.encode("ab"))


if __name__ == "__main__":
    unittest.main()

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
        transformer_config = config.to_transformer_config()
        self.assertEqual(transformer_config.num_layers, 1)
        self.assertEqual(transformer_config.position_embedding, "absolute")
        self.assertFalse(transformer_config.scale_attention_scores)

    def test_rejects_non_gpt_neo_config(self) -> None:
        with self.assertRaises(ValueError):
            NumpyGptNeoConfig.from_dict({"model_type": "llama"})


class NumpyGptNeoLanguageModelTest(unittest.TestCase):
    def test_runtime_skeleton_prefill_and_decode(self) -> None:
        np = require_numpy_for_test()

        from tiny_invoker.gpt_neo import NumpyGptNeoLanguageModel
        from tiny_invoker.interfaces import ForwardInput, ForwardMode
        from tiny_invoker.tokenizer import CharTokenizer
        from tiny_invoker.transformer import DecoderOnlyTransformer

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
                    "transformer.h.0.ln_1.weight": np.ones((3,), dtype=np.float32),
                    "transformer.h.0.ln_1.bias": np.zeros((3,), dtype=np.float32),
                    "transformer.h.0.ln_2.weight": np.ones((3,), dtype=np.float32),
                    "transformer.h.0.ln_2.bias": np.zeros((3,), dtype=np.float32),
                    "transformer.h.0.attn.attention.q_proj.weight": np.eye(3, dtype=np.float32),
                    "transformer.h.0.attn.attention.k_proj.weight": np.eye(3, dtype=np.float32),
                    "transformer.h.0.attn.attention.v_proj.weight": np.eye(3, dtype=np.float32),
                    "transformer.h.0.attn.attention.out_proj.weight": np.eye(3, dtype=np.float32),
                    "transformer.h.0.attn.attention.out_proj.bias": np.zeros((3,), dtype=np.float32),
                    "transformer.h.0.mlp.c_fc.weight": np.zeros((12, 3), dtype=np.float32),
                    "transformer.h.0.mlp.c_fc.bias": np.zeros((12,), dtype=np.float32),
                    "transformer.h.0.mlp.c_proj.weight": np.zeros((3, 12), dtype=np.float32),
                    "transformer.h.0.mlp.c_proj.bias": np.zeros((3,), dtype=np.float32),
                    "transformer.ln_f.weight": np.ones((3,), dtype=np.float32),
                    "transformer.ln_f.bias": np.zeros((3,), dtype=np.float32),
                },
            )

            model = NumpyGptNeoLanguageModel.from_files(
                config_path=config_path,
                weights_path=weights_path,
                tokenizer=tokenizer,
            )
            self.assertIsInstance(model.transformer, DecoderOnlyTransformer)
            self.assertEqual(
                model.transformer.weights.lm_head_weight_t.shape,
                (3, tokenizer.vocab_size),
            )
            self.assertIsNone(
                model.transformer._attention_mask(
                    query_length=1,
                    key_length=2,
                    start_position=1,
                    attention_type="global",
                )
            )
            local_decode_mask = model.transformer._attention_mask(
                query_length=1,
                key_length=11,
                start_position=10,
                attention_type="local",
            )
            profiled_prefill, profile = model.profile_forward(
                ForwardInput(
                    token_ids=tokenizer.encode("a"),
                    mode=ForwardMode.PREFILL,
                )
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

        self.assertEqual(
            local_decode_mask.tolist(),
            [[False, False, False, True, True, True, True, True, True, True, True]],
        )
        self.assertEqual(profiled_prefill.logits.shape, (tokenizer.vocab_size,))
        self.assertIn("attention_ms", profile)
        self.assertIn("mlp_ms", profile)
        self.assertIn("lm_head_ms", profile)
        self.assertEqual(len(prefill.logits), tokenizer.vocab_size)
        self.assertEqual(prefill.logits.shape, (tokenizer.vocab_size,))
        self.assertEqual(prefill.cache.token_ids, tokenizer.encode("a"))
        self.assertEqual(prefill.cache.length, 1)
        self.assertEqual(prefill.cache.capacity, 8)
        self.assertEqual(len(prefill.cache.keys), 1)
        self.assertEqual(prefill.cache.keys[0].shape, (1, 8, 3))
        self.assertEqual(prefill.cache.values[0].shape, (1, 8, 3))
        self.assertEqual(decode.cache.token_ids, tokenizer.encode("ab"))
        self.assertEqual(decode.cache.length, 2)
        self.assertEqual(decode.cache.capacity, 8)
        self.assertEqual(decode.cache.keys[0].shape, (1, 8, 3))
        self.assertEqual(decode.cache.values[0].shape, (1, 8, 3))
        self.assertIs(decode.cache.keys[0], prefill.cache.keys[0])
        self.assertIs(decode.cache.values[0], prefill.cache.values[0])


if __name__ == "__main__":
    unittest.main()

import unittest
from pathlib import Path

from tiny_invoker.hf import model_cache_dir, model_info_from_payload


class HfModelInfoTest(unittest.TestCase):
    def test_summarizes_tinystories_gpt_neo_config(self) -> None:
        api_payload = {
            "sha": "abc123",
            "siblings": [
                {"rfilename": "config.json"},
                {"rfilename": "merges.txt"},
                {"rfilename": "pytorch_model.bin"},
                {"rfilename": "tokenizer.json"},
                {"rfilename": "vocab.json"},
            ],
        }
        config_payload = {
            "architectures": ["GPTNeoForCausalLM"],
            "model_type": "gpt_neo",
            "num_layers": 4,
            "hidden_size": 768,
            "num_heads": 16,
            "vocab_size": 50257,
            "max_position_embeddings": 2048,
            "window_size": 256,
        }

        info = model_info_from_payload(
            "roneneldan/TinyStories-33M",
            api_payload=api_payload,
            config_payload=config_payload,
        )

        self.assertEqual(info.architecture, "GPTNeoForCausalLM")
        self.assertEqual(info.model_type, "gpt_neo")
        self.assertEqual(info.missing_files(), ())
        self.assertIn("hidden_size: 768", info.summary_lines())
        self.assertIn("required_files: ok", info.summary_lines())

    def test_reports_missing_required_files(self) -> None:
        info = model_info_from_payload(
            "example/model",
            api_payload={"siblings": [{"rfilename": "config.json"}]},
            config_payload={"architectures": ["GPTNeoForCausalLM"]},
        )

        self.assertEqual(
            info.missing_files(),
            ("tokenizer.json", "vocab.json", "merges.txt", "pytorch_model.bin"),
        )

    def test_builds_stable_model_cache_dir(self) -> None:
        path = model_cache_dir(
            "roneneldan/TinyStories-33M",
            revision="main",
            cache_dir=Path("/tmp/tiny-cache"),
        )

        self.assertEqual(
            path,
            Path("/tmp/tiny-cache/hf/roneneldan--TinyStories-33M/main"),
        )


if __name__ == "__main__":
    unittest.main()

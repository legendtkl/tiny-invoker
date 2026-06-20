import unittest
from pathlib import Path

from tiny_invoker.weights import manifest_from_state_dict


class FakeTensor:
    def __init__(self, shape: tuple[int, ...], dtype: str) -> None:
        self.shape = shape
        self.dtype = dtype

    def numel(self) -> int:
        total = 1
        for dimension in self.shape:
            total *= dimension
        return total


class WeightsTest(unittest.TestCase):
    def test_builds_manifest_from_state_dict(self) -> None:
        manifest = manifest_from_state_dict(
            Path("/tmp/model.bin"),
            {
                "transformer.wte.weight": FakeTensor((50257, 768), "float32"),
                "transformer.h.0.ln_1.weight": FakeTensor((768,), "float32"),
            },
        )

        self.assertEqual(manifest.total_tensors, 2)
        self.assertEqual(manifest.total_values, 50257 * 768 + 768)
        self.assertEqual(
            manifest.tensors[0].summary_line(),
            "transformer.h.0.ln_1.weight: shape=[768] dtype=float32 values=768",
        )

    def test_summary_lines_can_limit_tensor_output(self) -> None:
        manifest = manifest_from_state_dict(
            Path("/tmp/model.bin"),
            {
                "a": FakeTensor((2, 2), "float32"),
                "b": FakeTensor((3,), "float32"),
            },
        )

        lines = manifest.summary_lines(limit=1)

        self.assertIn("total_tensors: 2", lines)
        self.assertIn("... 1 more tensors", lines)


if __name__ == "__main__":
    unittest.main()

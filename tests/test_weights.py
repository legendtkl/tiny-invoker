import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tiny_invoker.weights import (
    manifest_from_state_dict,
    save_npz_weights,
    should_export_tensor,
    state_dict_to_numpy_arrays,
)


class FakeTensor:
    def __init__(self, shape: tuple[int, ...], dtype: str) -> None:
        self.shape = shape
        self.dtype = dtype

    def numel(self) -> int:
        total = 1
        for dimension in self.shape:
            total *= dimension
        return total


class FakeTorchTensor(FakeTensor):
    def __init__(self, shape: tuple[int, ...], dtype: str, value: int) -> None:
        super().__init__(shape, dtype)
        self.value = value

    def detach(self) -> "FakeTorchTensor":
        return self

    def cpu(self) -> "FakeTorchTensor":
        return self

    def numpy(self) -> list[int]:
        return [self.value]


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

    def test_filters_non_runtime_weight_buffers(self) -> None:
        self.assertFalse(should_export_tensor("transformer.h.0.attn.attention.bias", FakeTensor((1,), "torch.bool")))
        self.assertFalse(should_export_tensor("transformer.h.0.attn.attention.masked_bias", FakeTensor((), "torch.float32")))
        self.assertTrue(should_export_tensor("transformer.wte.weight", FakeTensor((2, 2), "torch.float32")))

    def test_converts_state_dict_to_numpy_arrays(self) -> None:
        arrays = state_dict_to_numpy_arrays(
            {
                "transformer.h.0.attn.attention.bias": FakeTorchTensor((1,), "torch.bool", 0),
                "transformer.wte.weight": FakeTorchTensor((2, 2), "torch.float32", 7),
            }
        )

        self.assertEqual(list(arrays), ["transformer.wte.weight"])
        self.assertEqual(arrays["transformer.wte.weight"], [7])

    def test_saves_npz_weights(self) -> None:
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("NumPy is only required for optional weight conversion.")

        with TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "weights.npz"
            manifest = save_npz_weights(
                output,
                {"a": [1, 2, 3]},
                compressed=False,
            )

            self.assertTrue(output.exists())
            self.assertEqual(manifest.total_tensors, 1)


if __name__ == "__main__":
    unittest.main()

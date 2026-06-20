from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class WeightTensorInfo:
    name: str
    shape: tuple[int, ...]
    dtype: str
    value_count: int

    def summary_line(self) -> str:
        shape = "[" + ", ".join(str(dimension) for dimension in self.shape) + "]"
        return (
            f"{self.name}: shape={shape} "
            f"dtype={self.dtype} values={self.value_count}"
        )


@dataclass(frozen=True)
class WeightManifest:
    path: Path
    tensors: tuple[WeightTensorInfo, ...]

    @property
    def total_tensors(self) -> int:
        return len(self.tensors)

    @property
    def total_values(self) -> int:
        return sum(tensor.value_count for tensor in self.tensors)

    def summary_lines(self, limit: int | None = 80) -> list[str]:
        lines = [
            f"weights_file: {self.path}",
            f"total_tensors: {self.total_tensors}",
            f"total_values: {self.total_values}",
        ]
        tensors = self.tensors if limit is None else self.tensors[:limit]
        lines.extend(tensor.summary_line() for tensor in tensors)
        if limit is not None and len(self.tensors) > limit:
            lines.append(f"... {len(self.tensors) - limit} more tensors")
        return lines


def tensor_info_from_state_dict(
    state_dict: Mapping[str, Any],
) -> tuple[WeightTensorInfo, ...]:
    tensors: list[WeightTensorInfo] = []
    for name in sorted(state_dict):
        value = state_dict[name]
        if not hasattr(value, "shape") or not hasattr(value, "dtype"):
            continue

        shape = tuple(int(dimension) for dimension in value.shape)
        if hasattr(value, "numel"):
            value_count = int(value.numel())
        else:
            value_count = 1
            for dimension in shape:
                value_count *= dimension
        tensors.append(
            WeightTensorInfo(
                name=name,
                shape=shape,
                dtype=str(value.dtype),
                value_count=value_count,
            )
        )
    return tuple(tensors)


def manifest_from_state_dict(
    path: str | Path,
    state_dict: Mapping[str, Any],
) -> WeightManifest:
    return WeightManifest(
        path=Path(path),
        tensors=tensor_info_from_state_dict(state_dict),
    )


def load_torch_weight_manifest(path: str | Path) -> WeightManifest:
    state = load_torch_state_dict(path)
    return manifest_from_state_dict(path, state)


def load_torch_state_dict(path: str | Path) -> Mapping[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch weight loading requires the optional weights dependencies. "
            "Install them with `python3 -m pip install '.[weights]'` from this repository."
        ) from error

    state = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=True,
    )
    if isinstance(state, Mapping) and "state_dict" in state and isinstance(state["state_dict"], Mapping):
        state = state["state_dict"]
    if not isinstance(state, Mapping):
        raise ValueError("Expected a PyTorch state_dict mapping.")
    return state


def should_export_tensor(name: str, value: Any) -> bool:
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        return False
    dtype = str(value.dtype)
    if dtype in {"torch.bool", "bool"}:
        return False
    if name.endswith(".attention.bias") or name.endswith(".attention.masked_bias"):
        return False
    return True


def state_dict_to_numpy_arrays(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    arrays: dict[str, Any] = {}
    for name in sorted(state_dict):
        value = state_dict[name]
        if not should_export_tensor(name, value):
            continue
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        arrays[name] = value
    return arrays


def save_npz_weights(
    output_path: str | Path,
    arrays: Mapping[str, Any],
    compressed: bool = True,
) -> WeightManifest:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "NPZ weight conversion requires NumPy. Install optional dependencies with "
            "`python3 -m pip install '.[weights]'` from this repository."
        ) from error

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    array_payload = {name: np.asarray(value) for name, value in arrays.items()}
    if compressed:
        np.savez_compressed(output, **array_payload)
    else:
        np.savez(output, **array_payload)
    return manifest_from_state_dict(output, array_payload)


def convert_torch_weights_to_npz(
    input_path: str | Path,
    output_path: str | Path,
    compressed: bool = True,
) -> WeightManifest:
    state_dict = load_torch_state_dict(input_path)
    arrays = state_dict_to_numpy_arrays(state_dict)
    return save_npz_weights(output_path, arrays, compressed=compressed)

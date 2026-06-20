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
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "inspect-weights requires PyTorch. Install it with "
            "`python3 -m pip install '.[weights]'` from this repository."
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
    return manifest_from_state_dict(path, state)

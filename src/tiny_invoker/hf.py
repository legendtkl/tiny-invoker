from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote
from urllib.request import urlopen


DEFAULT_HF_ENDPOINT = "https://huggingface.co"
GPT_NEO_REQUIRED_FILES = (
    "config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "pytorch_model.bin",
)


@dataclass(frozen=True)
class HfModelInfo:
    model_id: str
    sha: str | None
    config: dict[str, Any]
    files: tuple[str, ...]

    @property
    def architecture(self) -> str:
        architectures = self.config.get("architectures")
        if isinstance(architectures, list) and architectures:
            return str(architectures[0])
        return "unknown"

    @property
    def model_type(self) -> str:
        return str(self.config.get("model_type", "unknown"))

    def config_value(self, name: str) -> Any:
        return self.config.get(name)

    def missing_files(self, required_files: tuple[str, ...] = GPT_NEO_REQUIRED_FILES) -> tuple[str, ...]:
        available = set(self.files)
        return tuple(filename for filename in required_files if filename not in available)

    def summary_lines(self) -> list[str]:
        fields = [
            ("model_id", self.model_id),
            ("sha", self.sha or "unknown"),
            ("architecture", self.architecture),
            ("model_type", self.model_type),
            ("num_layers", self.config_value("num_layers")),
            ("hidden_size", self.config_value("hidden_size")),
            ("num_heads", self.config_value("num_heads")),
            ("vocab_size", self.config_value("vocab_size")),
            ("max_position_embeddings", self.config_value("max_position_embeddings")),
            ("window_size", self.config_value("window_size")),
        ]
        lines = [f"{name}: {value}" for name, value in fields if value is not None]
        missing = self.missing_files()
        if missing:
            lines.append("missing_files: " + ", ".join(missing))
        else:
            lines.append("required_files: ok")
        return lines


def model_info_from_payload(
    model_id: str,
    api_payload: dict[str, Any],
    config_payload: dict[str, Any],
) -> HfModelInfo:
    siblings = api_payload.get("siblings", [])
    files: list[str] = []
    if isinstance(siblings, list):
        for sibling in siblings:
            if isinstance(sibling, dict) and isinstance(sibling.get("rfilename"), str):
                files.append(sibling["rfilename"])

    sha = api_payload.get("sha")
    return HfModelInfo(
        model_id=model_id,
        sha=str(sha) if sha is not None else None,
        config=config_payload,
        files=tuple(sorted(files)),
    )


def fetch_json(url: str, timeout: float = 20.0) -> dict[str, Any]:
    with urlopen(url, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object from {url}.")
    return payload


def fetch_model_info(
    model_id: str,
    endpoint: str = DEFAULT_HF_ENDPOINT,
    revision: str = "main",
    timeout: float = 20.0,
) -> HfModelInfo:
    quoted_model_id = quote(model_id, safe="/")
    quoted_revision = quote(revision, safe="")
    api_payload = fetch_json(f"{endpoint}/api/models/{quoted_model_id}", timeout=timeout)
    config_payload = fetch_json(
        f"{endpoint}/{quoted_model_id}/raw/{quoted_revision}/config.json",
        timeout=timeout,
    )
    return model_info_from_payload(
        model_id=model_id,
        api_payload=api_payload,
        config_payload=config_payload,
    )

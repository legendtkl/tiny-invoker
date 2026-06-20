from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from pathlib import Path
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


def default_cache_dir() -> Path:
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Caches" / "tiny-invoker"
    return Path.home() / ".cache" / "tiny-invoker"


def model_cache_dir(
    model_id: str,
    revision: str = "main",
    cache_dir: Path | None = None,
) -> Path:
    root = cache_dir or default_cache_dir()
    safe_model_id = model_id.replace("/", "--")
    safe_revision = revision.replace("/", "--")
    return root / "hf" / safe_model_id / safe_revision


def download_model_file(
    model_id: str,
    filename: str,
    endpoint: str = DEFAULT_HF_ENDPOINT,
    revision: str = "main",
    cache_dir: Path | None = None,
    timeout: float = 60.0,
) -> Path:
    target_dir = model_cache_dir(model_id, revision=revision, cache_dir=cache_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename
    if target_path.exists():
        return target_path

    quoted_model_id = quote(model_id, safe="/")
    quoted_revision = quote(revision, safe="")
    quoted_filename = quote(filename, safe="/")
    url = f"{endpoint}/{quoted_model_id}/resolve/{quoted_revision}/{quoted_filename}"

    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    with urlopen(url, timeout=timeout) as response:
        with tmp_path.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
    tmp_path.replace(target_path)
    return target_path

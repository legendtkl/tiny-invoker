from __future__ import annotations

import json
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from tiny_invoker.engine import GenerationConfig, InferenceEngine


@dataclass(frozen=True)
class GenerateRequest:
    prompt: str
    max_new_tokens: int = 20
    temperature: float = 0.8
    top_k: int | None = 20
    seed: int | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "GenerateRequest":
        prompt = payload.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("Field 'prompt' must be a string.")

        max_new_tokens = payload.get("max_new_tokens", 20)
        if not isinstance(max_new_tokens, int):
            raise ValueError("Field 'max_new_tokens' must be an integer.")

        temperature = payload.get("temperature", 0.8)
        if not isinstance(temperature, int | float):
            raise ValueError("Field 'temperature' must be a number.")

        top_k = payload.get("top_k", 20)
        if top_k is not None and not isinstance(top_k, int):
            raise ValueError("Field 'top_k' must be an integer or null.")

        seed = payload.get("seed")
        if seed is not None and not isinstance(seed, int):
            raise ValueError("Field 'seed' must be an integer or null.")

        request = cls(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=float(temperature),
            top_k=top_k,
            seed=seed,
        )
        request.to_generation_config().validate()
        return request

    def to_generation_config(self) -> GenerationConfig:
        return GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            seed=self.seed,
            trace=False,
        )


class TinyInvokerServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        engine: InferenceEngine,
        model_name: str,
    ) -> None:
        super().__init__(server_address, TinyInvokerHandler)
        self.engine = engine
        self.model_name = model_name


class TinyInvokerHandler(BaseHTTPRequestHandler):
    server: TinyInvokerServer

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/health":
            self._write_json(
                200,
                {
                    "status": "ok",
                    "model": self.server.model_name,
                },
            )
            return
        self._write_json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        if self.path != "/generate":
            self._write_json(404, {"error": "not_found"})
            return

        try:
            payload = self._read_json()
            request = GenerateRequest.from_payload(payload)
            response = self._generate(request)
        except ValueError as error:
            self._write_json(400, {"error": str(error)})
            return
        except Exception as error:
            self._write_json(500, {"error": f"{type(error).__name__}: {error}"})
            return

        self._write_json(200, response)

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("Request body must be valid JSON.") from error
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object.")
        return payload

    def _generate(self, request: GenerateRequest) -> dict[str, Any]:
        start = time.perf_counter()
        prompt_token_ids = self.server.engine.model.tokenizer.encode(request.prompt)
        result = self.server.engine.generate(
            request.prompt,
            config=request.to_generation_config(),
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        generated_tokens = len(result.token_ids) - len(prompt_token_ids)
        return {
            "text": result.text,
            "token_ids": result.token_ids,
            "usage": {
                "prompt_tokens": len(prompt_token_ids),
                "generated_tokens": generated_tokens,
                "total_tokens": len(result.token_ids),
            },
            "timing": {
                "elapsed_ms": round(elapsed_ms, 3),
                "tokens_per_second": round(generated_tokens / (elapsed_ms / 1000.0), 3)
                if elapsed_ms > 0 and generated_tokens > 0
                else 0.0,
            },
        }

    def _write_json(self, status_code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(
    engine: InferenceEngine,
    model_name: str,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    server = TinyInvokerServer((host, port), engine=engine, model_name=model_name)
    try:
        print(f"Serving {model_name} at http://{host}:{port}")
        print("POST /generate")
        print("GET  /health")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
    finally:
        server.server_close()

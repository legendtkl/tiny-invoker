import json
import threading
import unittest
from http.client import HTTPConnection

from tiny_invoker.engine import GenerationResult
from tiny_invoker.server import GenerateRequest, TinyInvokerServer
from tiny_invoker.tokenizer import CharTokenizer


class FakeEngine:
    def __init__(self) -> None:
        self.model = type("FakeModel", (), {"tokenizer": CharTokenizer.from_text("abc ")})()

    def generate(self, prompt: str, config) -> GenerationResult:
        generated = "a" * config.max_new_tokens
        token_ids = self.model.tokenizer.encode(prompt + generated)
        return GenerationResult(text=prompt + generated, token_ids=token_ids, steps=[])


class ServerTest(unittest.TestCase):
    def test_generate_request_validates_payload(self) -> None:
        request = GenerateRequest.from_payload(
            {
                "prompt": "abc",
                "max_new_tokens": 2,
                "temperature": 0,
                "top_k": None,
                "seed": 1,
            }
        )

        self.assertEqual(request.prompt, "abc")
        self.assertEqual(request.to_generation_config().max_new_tokens, 2)

    def test_generate_request_rejects_bad_payload(self) -> None:
        with self.assertRaises(ValueError):
            GenerateRequest.from_payload({"prompt": 123})

    def test_http_generate_endpoint(self) -> None:
        server = TinyInvokerServer(("127.0.0.1", 0), engine=FakeEngine(), model_name="fake")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            conn = HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
            conn.request(
                "POST",
                "/generate",
                body=json.dumps({"prompt": "bc", "max_new_tokens": 2}),
                headers={"Content-Type": "application/json"},
            )
            response = conn.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        self.assertEqual(response.status, 200)
        self.assertEqual(payload["text"], "bcaa")
        self.assertEqual(payload["usage"]["generated_tokens"], 2)


if __name__ == "__main__":
    unittest.main()

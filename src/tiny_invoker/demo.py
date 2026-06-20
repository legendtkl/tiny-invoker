from __future__ import annotations

from tiny_invoker.engine import InferenceEngine
from tiny_invoker.model import BigramLanguageModel
from tiny_invoker.tokenizer import CharTokenizer


DEMO_CORPUS = """
tiny invoker turns tokens into text.
the engine asks a model for the next token.
the sampler chooses one token and appends it.
small models are useful for learning.
learning inference starts with token ids.
模型推理从 token 开始。
学习推理引擎，可以先看清楚每一步。
模型根据上一个字符猜下一个字符。
"""

def build_demo_engine() -> InferenceEngine:
    tokenizer = CharTokenizer.from_text(DEMO_CORPUS)
    model = BigramLanguageModel.from_corpus(DEMO_CORPUS, tokenizer=tokenizer)
    return InferenceEngine(model=model)

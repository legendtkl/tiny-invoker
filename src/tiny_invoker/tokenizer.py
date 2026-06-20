from __future__ import annotations

from dataclasses import dataclass


BOS_TOKEN = "<bos>"
UNK_TOKEN = "<unk>"


@dataclass
class CharTokenizer:
    """A tiny character tokenizer.

    Real LLM tokenizers usually operate on subword units. This project starts
    with characters because the ids are easy to inspect while learning.
    """

    id_to_token: tuple[str, ...]
    token_to_id: dict[str, int]
    bos_token: str = BOS_TOKEN
    unk_token: str = UNK_TOKEN

    @classmethod
    def from_text(cls, text: str, extra_characters: str = "") -> "CharTokenizer":
        tokens = [BOS_TOKEN, UNK_TOKEN]
        for character in sorted(set(text + extra_characters)):
            if character not in tokens:
                tokens.append(character)
        return cls.from_tokens(tokens)

    @classmethod
    def from_tokens(cls, tokens: list[str]) -> "CharTokenizer":
        if BOS_TOKEN not in tokens:
            tokens.insert(0, BOS_TOKEN)
        if UNK_TOKEN not in tokens:
            tokens.insert(1, UNK_TOKEN)

        id_to_token = tuple(tokens)
        token_to_id = {token: token_id for token_id, token in enumerate(id_to_token)}
        if len(token_to_id) != len(id_to_token):
            raise ValueError("Tokenizer vocabulary contains duplicate tokens.")
        return cls(id_to_token=id_to_token, token_to_id=token_to_id)

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.bos_token]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.unk_token]

    @property
    def special_token_ids(self) -> set[int]:
        return {self.bos_id, self.unk_id}

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def encode(self, text: str) -> list[int]:
        return [self.token_to_id.get(character, self.unk_id) for character in text]

    def decode(self, token_ids: list[int]) -> str:
        pieces: list[str] = []
        for token_id in token_ids:
            token = self.id_to_token[token_id]
            if token == self.bos_token:
                continue
            if token == self.unk_token:
                pieces.append("?")
            else:
                pieces.append(token)
        return "".join(pieces)

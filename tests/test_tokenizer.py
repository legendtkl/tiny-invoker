import unittest
from tempfile import TemporaryDirectory

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel

from tiny_invoker.tokenizer import CharTokenizer, GPT2_END_OF_TEXT, HfTokenizer


class CharTokenizerTest(unittest.TestCase):
    def test_round_trips_known_characters(self) -> None:
        tokenizer = CharTokenizer.from_text("abc")

        token_ids = tokenizer.encode("cab")

        self.assertEqual(tokenizer.decode(token_ids), "cab")

    def test_unknown_character_decodes_to_question_mark(self) -> None:
        tokenizer = CharTokenizer.from_text("abc")

        token_ids = tokenizer.encode("z")

        self.assertEqual(token_ids, [tokenizer.unk_id])
        self.assertEqual(tokenizer.decode(token_ids), "?")


class HfTokenizerTest(unittest.TestCase):
    def test_loads_tokenizer_json_and_round_trips_text(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tokenizer_file = f"{tmp_dir}/tokenizer.json"
            raw_tokenizer = Tokenizer(
                BPE(
                    vocab={
                        "h": 0,
                        "i": 1,
                        "Ġ": 2,
                        GPT2_END_OF_TEXT: 3,
                    },
                    merges=[],
                )
            )
            raw_tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
            raw_tokenizer.decoder = ByteLevelDecoder()
            raw_tokenizer.add_special_tokens([GPT2_END_OF_TEXT])
            raw_tokenizer.save(tokenizer_file)

            tokenizer = HfTokenizer.from_file(tokenizer_file)

            self.assertEqual(tokenizer.vocab_size, 4)
            self.assertEqual(tokenizer.bos_id, 3)
            self.assertEqual(tokenizer.special_token_ids, {3})
            self.assertEqual(tokenizer.decode(tokenizer.encode("hi")), "hi")


if __name__ == "__main__":
    unittest.main()

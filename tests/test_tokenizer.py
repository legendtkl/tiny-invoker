import unittest

from tiny_invoker.tokenizer import CharTokenizer


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


if __name__ == "__main__":
    unittest.main()

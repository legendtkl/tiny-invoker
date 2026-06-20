import random
import unittest

from tiny_invoker.sampler import choose_token, filter_top_k, softmax, top_candidates


class SamplerTest(unittest.TestCase):
    def test_softmax_returns_probabilities_that_sum_to_one(self) -> None:
        probabilities = softmax([1.0, 2.0, 3.0])

        self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertGreater(probabilities[2], probabilities[1])

    def test_top_k_masks_lower_ranked_logits(self) -> None:
        filtered = filter_top_k([1.0, 3.0, 2.0], top_k=2)

        self.assertEqual(filtered[0], float("-inf"))
        self.assertEqual(filtered[1], 3.0)
        self.assertEqual(filtered[2], 2.0)

    def test_zero_temperature_chooses_highest_logit(self) -> None:
        token_id = choose_token([1.0, 3.0, 2.0], rng=random.Random(0), temperature=0)

        self.assertEqual(token_id, 1)

    def test_top_candidates_are_sorted(self) -> None:
        candidates = top_candidates([1.0, 3.0, 2.0], top_n=2)

        self.assertEqual([candidate.token_id for candidate in candidates], [1, 2])

    def test_top_candidates_omit_masked_logits(self) -> None:
        candidates = top_candidates([float("-inf"), 3.0, 2.0], top_n=3)

        self.assertEqual([candidate.token_id for candidate in candidates], [1, 2])


if __name__ == "__main__":
    unittest.main()

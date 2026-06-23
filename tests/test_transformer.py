import unittest


def require_numpy_for_test():
    try:
        import numpy as np
    except ImportError:
        raise unittest.SkipTest("NumPy is only required for optional Transformer tests.")
    return np


class GroupedQueryAttentionTest(unittest.TestCase):
    def test_kv_cache_allocates_small_capacity_and_grows(self) -> None:
        np = require_numpy_for_test()

        from tiny_invoker.transformer import (
            DecoderOnlyTransformer,
            TransformerConfig,
            TransformerWeights,
        )

        transformer = DecoderOnlyTransformer(
            config=TransformerConfig(
                vocab_size=4,
                hidden_size=2,
                max_position_embeddings=64,
                layer_norm_epsilon=1e-5,
                num_layers=0,
                num_heads=1,
                intermediate_size=4,
                attention_layers=(),
                window_size=64,
            ),
            weights=TransformerWeights(
                token_embedding=np.zeros((4, 2), dtype=np.float32),
                position_embedding=None,
                layers=(),
                final_norm_weight=np.ones((2,), dtype=np.float32),
                final_norm_bias=None,
            ),
        )

        first_token = np.ones((1, 1, 2), dtype=np.float32)
        cache = transformer._updated_kv_cache(
            past_cache=None,
            new_values=first_token,
            start_position=0,
            end_position=1,
        )
        self.assertEqual(cache.shape, (1, 16, 2))

        next_tokens = np.ones((1, 16, 2), dtype=np.float32) * 2.0
        grown_cache = transformer._updated_kv_cache(
            past_cache=cache,
            new_values=next_tokens,
            start_position=1,
            end_position=17,
        )

        self.assertEqual(grown_cache.shape, (1, 32, 2))
        np.testing.assert_allclose(grown_cache[:, :1, :], first_token)
        np.testing.assert_allclose(grown_cache[:, 1:17, :], next_tokens)

    def test_grouped_attention_matches_repeated_kv_heads(self) -> None:
        np = require_numpy_for_test()

        from tiny_invoker.transformer import (
            grouped_query_attention_context,
            grouped_query_attention_scores,
            repeat_key_value_heads,
            softmax,
        )

        group_size = 2
        head_dim = 3
        query = np.arange(4 * 2 * head_dim, dtype=np.float32).reshape(4, 2, head_dim) / 10.0
        key = np.arange(2 * 5 * head_dim, dtype=np.float32).reshape(2, 5, head_dim) / 20.0
        value = np.arange(2 * 5 * head_dim, dtype=np.float32).reshape(2, 5, head_dim) / 30.0

        repeated_key = repeat_key_value_heads(key, group_size)
        repeated_value = repeat_key_value_heads(value, group_size)
        repeated_scores = query @ np.swapaxes(repeated_key, -1, -2) / np.sqrt(head_dim)
        repeated_probabilities = softmax(repeated_scores, axis=-1)
        repeated_context = repeated_probabilities @ repeated_value

        grouped_query = query.reshape(2, group_size, 2, head_dim)
        grouped_scores = grouped_query_attention_scores(
            query=grouped_query,
            key=key,
            group_size=group_size,
            scale_attention_scores=True,
            head_dim=head_dim,
        )
        grouped_probabilities = softmax(grouped_scores, axis=-1)
        grouped_context = grouped_query_attention_context(
            probabilities=grouped_probabilities,
            value=value,
            group_size=group_size,
        )

        np.testing.assert_allclose(grouped_scores, repeated_scores)
        np.testing.assert_allclose(grouped_context, repeated_context)


if __name__ == "__main__":
    unittest.main()

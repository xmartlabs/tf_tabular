import tensorflow as tf


class SequenceProcessor:
    def __init__(
        self, combine_before_attention: bool = True, attn_heads: int = 4, key_dim: int = 256, attention_builder=None
    ):
        """_summary_

        :param bool combine_before_attention: Whether to combine sequencial inputs before attention, defaults to True. Only True is supported right now.
        :param int attn_heads: _description_, defaults to 4
        :param int key_dim: _description_, defaults to 256
        :param _type_ attention_builder: _description_, defaults to None
        """
        self.combine_before_attention = combine_before_attention
        self.attn_heads = attn_heads
        self.key_dim = key_dim
        self.attention_builder = attention_builder

    def combine(self, x):
        # if self.combine_before_attention:
        # Make sure numerical elements can be concatenated to embeddings
        reshaped = [t if t.rank == 3 else tf.expand_dims(t, axis=-1) for t in x]
        return tf.keras.layers.Concatenate(axis=-1)(reshaped)

    def attention(self, x):
        if self.attention_builder is not None:
            return self.attention_builder(x)
        return tf.keras.layers.MultiHeadAttention(num_heads=self.attn_heads, key_dim=self.key_dim)(x, x)

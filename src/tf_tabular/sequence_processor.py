from typing import List
import tensorflow as tf


class SequenceProcessor:
    def __init__(self, attn_heads: int = 4, key_dim: int = 256, attention_builder=None):
        """The SequenceProcessor concatenates sequential input layers and applies attention on top.

        :param int attn_heads: How many attention heads to use, defaults to 4
        :param int key_dim: key_dim passed to attention layer, defaults to 256
        :param function attention_builder: Optional function that takes a tf.keras.Layer and builds an attention or
        whatever other layer on top., defaults to None
        """
        self.attn_heads = attn_heads
        self.key_dim = key_dim
        self.attention_builder = attention_builder

    def _combine(self, x: List[tf.keras.Layer]):
        # Make sure numerical elements can be concatenated to embeddings
        reshaped = [t if len(t.shape) == 3 else tf.expand_dims(t, axis=-1) for t in x]
        return tf.keras.layers.Concatenate(axis=-1)(reshaped)

    def _attention(self, x: tf.keras.Layer):
        if self.attention_builder is not None:
            return self.attention_builder(x)
        return tf.keras.layers.MultiHeadAttention(num_heads=self.attn_heads, key_dim=self.key_dim)(x, x)

    def process_layers(self, x: List[tf.keras.Layer]):
        """Processes a list of layers, concatenating them and applying an attention layer."""
        x = self._combine(x)
        x = self._attention(x)
        return x

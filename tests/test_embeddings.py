import numpy as np
import pandas as pd
import tensorflow as tf

from tf_tabular.utils import get_embedding_matrix, batch_run_lookup_on_df


def test_batch_run_lookup_on_df():
    lookup = tf.keras.layers.StringLookup()
    df = pd.DataFrame({"id": ["b", "c", "e", "d", "f", "g", "h", "i", "j", "k"]})
    lookup.adapt(df.id.values)
    df.loc[df.shape[0]] = ["a"]
    out = batch_run_lookup_on_df(df, lookup, batch_size=3)
    expected = np.array([lookup(x) for x in df.id.values])
    assert np.array_equal(out, expected)


def test_get_embedding_matrix():
    lookup = tf.keras.layers.StringLookup()
    df = pd.DataFrame({"id": ["b", "c", "e", "d", "f", "g", "h", "i", "j", "k"]})
    lookup.adapt(df.id.values)
    df.loc[df.shape[0]] = ["a"]
    df = df.assign(embedding=df.id.apply(lambda x: np.array([ord(x)])))
    matrix = get_embedding_matrix(lookup, df)
    indexes = [(x, lookup(x)) for x in df.id.values]
    assert matrix.shape == (len(df), 1)
    for index in indexes:
        assert np.array_equal(matrix[index[1]], df[df.id == index[0]].embedding.values[0])


def test_get_embedding_matrix_multi_hot():
    lookup = tf.keras.layers.StringLookup(output_mode="multi_hot")
    df = pd.DataFrame({"id": [["a", "b", "c"], ["d", "e"], ["f", "g", "h", "i", "j"]]})
    vocab = df.id.explode().unique()

    lookup.adapt(vocab)
    df.loc[df.shape[0]] = [["k"]]
    vocab = list(vocab)
    vocab.extend(["k"])
    df = pd.DataFrame(df.id.explode().reset_index(drop=True))
    df = df.assign(embedding=[np.array([ord(x)]) for x in vocab])
    matrix = get_embedding_matrix(lookup, df)
    assert lookup.output_mode == "multi_hot"
    assert matrix.shape == (len(vocab), 1)

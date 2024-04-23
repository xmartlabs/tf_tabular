from itertools import combinations
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (
    StringLookup,
    IntegerLookup,
    Embedding,
    Normalization,
    GlobalAveragePooling1D,
    Reshape,
    Dense,
    GlobalMaxPool1D,
    Concatenate,
)
from tensorflow.keras.regularizers import L2


def _input_layer(name, is_multi_hot=False, is_string=False):
    if is_multi_hot:
        shape = (None,)
        # ragged = True
    else:
        shape = (1,)
        # ragged = False
    dtype = tf.string if is_string else None
    return tf.keras.Input(shape=shape, dtype=dtype, name=name)  # , ragged=ragged


def get_combiner(combiner, is_multi_hot):
    if not is_multi_hot:
        return Reshape((-1,))
    elif combiner == "mean":
        return GlobalAveragePooling1D()
    elif combiner == "sum":
        return tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))
    elif combiner == "max":
        return GlobalMaxPool1D()
    else:
        raise ValueError(f"Unknown combiner: {combiner}")


def build_projection_layer(cont_layers, num_projection, l2_reg, cross_features=True):
    if cross_features:
        cont_layers = list(cont_layers)
        pairs = list(combinations(cont_layers, 2))
        for p in pairs:
            cont_layers.append(tf.math.multiply(p[0], p[1]))
    concat = Concatenate(axis=1, name="concat_continuous")(cont_layers)
    return Dense(num_projection, activation="relu", name="continuous_projection", kernel_regularizer=L2(l2_reg))(concat)


def batch_run_lookup_on_df(df, lookup, batch_size=1000):
    """Expects a lookup layer and a dataframe with a column 'id' containing the strings to lookup."""
    out = []
    for i in range(0, len(df), batch_size):
        out.append(lookup(df.iloc[i : i + batch_size].id.values))
    return np.concatenate(out)


def get_embedding_matrix(lookup, embedding_df: pd.DataFrame):
    """Expects a lookup layer and a dataframe with columns 'id' and 'embedding' where the embeddings are already
    converted to numpy arrays.
    """
    om = lookup.output_mode
    lookup.output_mode = "int"
    out_of_vocab = embedding_df[~embedding_df.id.isin(lookup.get_vocabulary())]
    embedding_df = embedding_df[embedding_df.id.isin(lookup.get_vocabulary())]
    embedding_df["vocab_id"] = batch_run_lookup_on_df(embedding_df, lookup)
    embedding_df = embedding_df.sort_values("vocab_id")
    matrix = np.stack(embedding_df.embedding.values).astype(np.float32)
    oov_embedding = np.mean(np.stack(out_of_vocab.embedding.values), axis=0).reshape(1, -1)
    matrix = np.concatenate([oov_embedding, matrix], axis=0)
    lookup.output_mode = om
    return matrix


def get_embedding_layer(num_tokens, embedding_dim, name, lookup, embedding_df=None, verbose=False):
    if embedding_df is None:
        return Embedding(num_tokens, embedding_dim, name=name)
    embedding_matrix = get_embedding_matrix(lookup, embedding_df)
    if verbose:
        print("Num tokens:", num_tokens, ", embedding matrix:", embedding_matrix.shape)
        print("Size of the matrix: ", embedding_matrix.size)
        print("Memory size of one array element in bytes: ", embedding_matrix.itemsize)
        print("Total size in kb: ", embedding_matrix.itemsize * embedding_matrix.size / 1024)

    return Embedding(
        num_tokens,
        embedding_matrix.shape[1],
        name=name,
        embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
        trainable=False,
    )


def build_continuous_input(name, mean: float | None = None, variance: float | None = None, sample=None):
    """Build th e input layer stack for continuous features

    :param str name: Layer name
    :param float mean: mean of the feature values, defaults to None
    :param float variance: variance of the feature values, defaults to None
    :param _type_ sample: A sample of features to adapt the layer. You must specify either mean + variance or sample, not both
    :return tuple: preprocessed input and inputs
    """
    inp = _input_layer(name)
    norm_layer = Normalization(axis=None, name=name + "_norm", mean=mean, variance=variance)
    if sample is not None:
        norm_layer.adapt(sample)
    return (norm_layer(inp), inp)


def build_categorical_input(name, embedding_dim, vocab, is_multi_hot, embedding_df=None):
    """Build input for categorical columns.
    This function supports many cases because of the different trials we have done for different columns

    :param str name: Layer name
    :param int embedding_dim: Output dimension of embedding layer (ignored if embedding_df is not None)
    :param List[str | int] or similar, vocab: Vocabulary for lookup layer
    :param str is_multi_hot: Whether the column is a multi-hot
    :param dict embedding_df: Dict mapping columns to DataFrame containing precomputed embeddings, defaults to None

    :return tuple: preprocessed input and inputs
    """
    is_string = isinstance(vocab[0], (np.bytes_, bytes, str))

    # Choose correct lookup
    lookup_type = StringLookup if is_string else IntegerLookup

    # Build layers
    inp = _input_layer(name, is_multi_hot, is_string)
    if is_string and isinstance(type(vocab[0]), str):
        vocab = [a.encode() for a in vocab]
    lookup = lookup_type(vocabulary=vocab, output_mode="int", name="lookup_" + name)
    x = lookup(inp)

    # Used for all categoricals but specially for columns which use custom precomputed embeddings
    x = get_embedding_layer(
        len(vocab) + 1, embedding_dim, name=name + "_emb", lookup=lookup, embedding_df=embedding_df
    )(x)

    return (x, inp)


def get_vocab(series, max_size):
    if isinstance(series.iloc[0], list) or isinstance(series.iloc[0], np.ndarray):
        series = series.explode()
    series = series.dropna()
    uniques, counts = np.unique(series, return_counts=True)
    if max_size is not None:
        uniques = uniques[np.argsort(counts)[-max_size:]]
    vocab = set(uniques)
    if "_none_" in vocab:
        vocab.remove("_none_")
    vocab = sorted(vocab)
    return vocab

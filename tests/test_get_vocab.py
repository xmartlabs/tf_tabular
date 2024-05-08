import pandas as pd
from tf_tabular.utils import get_vocab


def test_get_vocab_string():
    df = pd.Series(["a", "b", "c", "a", "b", "c"])
    vocab = get_vocab(df)
    assert set(vocab) == set(["a", "b", "c"])


def test_get_vocab_max_size():
    df = pd.Series(["a", "b", "c", "a", "b"])
    vocab = get_vocab(df, max_size=2)
    assert set(vocab) == set(["a", "b"])


def test_get_vocab_int():
    df = pd.Series([1, 2, 3, 1, 2, 3])
    vocab = get_vocab(df)
    assert set(vocab) == set([1, 2, 3])


def test_exclude_none():
    df = pd.Series(["a", "b", "_none_"])
    vocab = get_vocab(df)
    assert set(vocab) == set(["a", "b"])


def test_vocab_lists():
    df = pd.Series([["a", "b"], ["c", "d"], ["a", "b"], ["c", "b"]])
    vocab = get_vocab(df)
    assert set(vocab) == set(["a", "b", "c", "d"])

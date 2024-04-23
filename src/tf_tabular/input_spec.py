from enum import Enum
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List
from .utils import build_continuous_input, build_categorical_input


class ColumnType(Enum):
    NUMERIC = 1
    CATEGORICAL = 2


class InputSpec:
    input_layers: List[tf.keras.layers.Layer] = []
    output_layer: tf.keras.layers.Layer | None = None

    def __init__(self, name: str, column_type: ColumnType, is_sequence: bool = False):
        self.name = name
        self.column_type = column_type
        self.is_sequence = is_sequence

    def build_layer(self):
        raise NotImplementedError("Method implemented in subclass")


class CategoricalInputSpec(InputSpec):
    def __init__(
        self,
        name: str,
        vocab: List[str],
        embedding_dim: int,
        is_sequence: bool = False,
        is_multi_hot: bool = False,
        embedding_df: pd.DataFrame = None,
    ):
        super().__init__(name, ColumnType.CATEGORICAL, is_sequence=is_sequence)
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.is_multi_hot = is_multi_hot
        self.embedding_df = embedding_df

    def build_layer(self):
        output, input = build_categorical_input(
            self.name, self.embedding_dim, self.vocab, self.is_multi_hot, embedding_df=self.embedding_df
        )
        self.input_layer = input
        self.output_layer = output


class NumInputSpec(InputSpec):
    sample: np.ndarray | None = None
    mean: float | None = None
    variance: float | None = None

    def __init__(self, name: str, norm_params: dict, is_sequence: bool = False):
        super().__init__(name, ColumnType.NUMERIC, is_sequence=is_sequence)
        if "sample" in norm_params:
            self.sample = norm_params["sample"]
            assert (
                "mean" not in norm_params and "var" not in norm_params
            ), "Sample and mean/variance cannot be specified together"
        elif "mean" in norm_params or "var" in norm_params:
            self.mean = norm_params["mean"]
            self.variance = norm_params["var"]
        else:
            raise NotImplementedError(
                "Only Standard normalization implemented. \
                                      Normalization parameters must contain 'mean' and 'var' or 'sample' keys"
            )

    def build_layer(self):
        output, input = build_continuous_input(self.name, self.mean, self.variance, self.sample)
        self.input_layer = input
        self.output_layer = output

from enum import Enum
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List
from .utils import build_continuous_input, build_categorical_input


logger = logging.getLogger(__name__)


class ColumnType(Enum):
    NUMERIC = 1
    CATEGORICAL = 2


class InputSpec:
    input_layer: tf.keras.layers.Layer | None = None
    output_layer: tf.keras.layers.Layer | None = None

    def __init__(self, name: str, column_type: ColumnType, is_sequence: bool = False):
        """Abstract class implemented by CategoricalInputSpec and NumInputSpec"""
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
        """CategoricalInputSpec handles the input specification for categorical features

        :param str name: name for the input layer
        :param List[str] vocab: List of unique values for the categorical feature
        :param int embedding_dim: Dimension of the embedding layer
        :param bool is_sequence: If the input is sequential, defaults to False
        :param bool is_multi_hot: If the input is multi_hot, defaults to False
        :param pd.DataFrame embedding_df: Dataframe containing 'id' and 'embeddings' columns with precomputed \
            embeddings, defaults to None
        """
        super().__init__(name, ColumnType.CATEGORICAL, is_sequence=is_sequence)
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.is_multi_hot = is_multi_hot
        self.embedding_df = embedding_df

    def build_layer(self):
        """Builds the input layer stack for the categorical feature"""
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
        """NumInputSpec handles the input specification for numerical features

        :param str name: name for the input layer
        :param dict norm_params: normalization parameters passed to the Normalization layer. If "sample" is present, \
            it will be used to adapt the layer. If "mean" and "var" are present, they will be used to normalize the layer.
            If none of these are present, no normalization will be applied.
        :param bool is_sequence: If the input is sequential, defaults to False
        """
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
            # No normalization
            logger.info(f"No normalization parameters found for {name}. Not normalizing this column")

    def build_layer(self):
        """Builds the input layer stack for the numeric feature"""
        output, input = build_continuous_input(self.name, self.mean, self.variance, self.sample)
        self.input_layer = input
        self.output_layer = output

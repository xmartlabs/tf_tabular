from typing import List, Dict
import tensorflow as tf
from .utils import get_combiner
from .sequence_processor import SequenceProcessor
from .numeric_processor import NumericProcessor
from .input_spec import ColumnType, InputSpec, NumInputSpec, CategoricalInputSpec


class InputBuilder:
    input_specs: List[InputSpec] = []

    def __init__(
        self,
        sequence_processor: SequenceProcessor | None = None,
        numeric_processor: NumericProcessor | None = None,
        combiner: str = "mean",
    ):
        """InputBuilder handles the input specification for the model

        :param SequenceProcessor | None sequence_processor: This will process sequential layers reducing their rank so \
            that they can be combined with other layers
        :param NumericProcessor | None numeric_processor: This will combine numeric layers, optionally adding a \
            projection layer on top
        :param str combiner: Used to define how to combine sequential features, defaults to "mean"
        """
        self.sequence_processor = sequence_processor
        self.combiner = combiner
        self.numeric_processor = numeric_processor or NumericProcessor()
        self.input_specs = []

    def add_inputs(self, input_specs: List[InputSpec]):
        """Adds input specifications to the model"""
        self.input_specs.extend(input_specs)

    def add_inputs_list(
        self,
        categoricals: List[str],
        numericals: List[str] = [],
        normalization_params: Dict = {},
        vocabs: Dict = {},
        embedding_dims: Dict = {},
        multi_hots: List[str] = [],
        sequentials: List[str] = [],
        embedding_df={},
    ):
        """Adds a list of columns to the input specification of the model.

        :param List[str] categoricals: List of categorical columns
        :param List[str] numericals: List of numerical columns
        :param Dict normalization_params: Dict mapping column name to normalization parameters
        :param Dict vocabs: Dict mapping column names to vocabularies
        :param Dict embedding_dims: Dict mapping column names to embedding dimensions (for categorical features)
        :param List[str] multi_hots: List of multi-hot columns
        :param List[str] sequentials: List of sequential columns
        :param dict embedding_df: Dict mapping column name to DataFrame containing precomputed embeddings
        """
        for cat in categoricals:
            self.input_specs.append(
                CategoricalInputSpec(
                    cat,
                    vocabs[cat],
                    embedding_dims[cat],
                    is_sequence=cat in sequentials,
                    is_multi_hot=cat in multi_hots,
                    embedding_df=embedding_df.get(cat, None),
                )
            )
        for num in numericals:
            self.input_specs.append(
                NumInputSpec(num, norm_params=normalization_params.get(num, {}), is_sequence=num in sequentials)
            )

    def build_input_layers(self) -> tuple[List[tf.keras.layers.Layer], tf.Tensor]:
        """Builds input layer stack and return the input layers and the output layer for building the model.
        :return tuple[List[tf.keras.layers.Layer], tf.Tensor]: Tuple containing the input layers and the output layer
        """
        input_layers = []
        sequence_layers = []
        output_layers = []
        projected_layers = []
        for spec in self.input_specs:
            spec.build_layer()
            input_layers.append(spec.input_layer)
            if spec.is_sequence:
                sequence_layers.append(spec.output_layer)
            elif spec.column_type == ColumnType.CATEGORICAL:
                output = self.merge_list(spec.output_layer, is_list=spec.is_multi_hot)  # type: ignore
                output_layers.append(output)
            else:  # Numeric layers
                projected_layers.append(spec.output_layer)

        if len(sequence_layers) > 0:
            if self.sequence_processor is None:
                raise ValueError("Sequence processor must be provided when specifying seuqence layers")
            x = self.sequence_processor.process_layers(sequence_layers)
            x = self.merge_list(x, is_list=True)
            output_layers.append(x)
        if len(projected_layers) > 0:
            if self.numeric_processor is None:
                raise ValueError("Numeric processor must be provided when specifying numeric layers")
            layers = self.numeric_processor.project(projected_layers)
            output_layers.extend(layers)
        output = tf.keras.layers.Concatenate(axis=-1)(output_layers)
        return input_layers, output

    def merge_list(self, x: tf.Tensor, is_list: bool):
        """Combines multi_hot of sequential layers reducing their rank to be combined with other layers.

        :param tf.keras.layers.Layer x: input layer
        :param bool is_list: if the input is a list or not
        :return EagerTensor: result of applying the combiner to the input
        """
        return get_combiner(self.combiner, is_list)(x)

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
        self.sequence_processor = sequence_processor
        self.combiner = combiner
        self.numeric_processor = numeric_processor or NumericProcessor()
        self.input_specs = []

    def add_inputs(self, input_specs: List[InputSpec]):
        self.input_specs.extend(input_specs)

    def add_inputs_list(
        self,
        categoricals,
        numericals: List[str] = [],
        normalization_params: Dict = {},
        vocabs: Dict = {},
        embedding_dims: Dict = {},
        multi_hots: List[str] = [],
        sequentials: List[str] = [],
        embedding_df={},
    ):
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
                NumInputSpec(num, norm_params=normalization_params[num], is_sequence=num in sequentials)
            )

    def build_input_layers(self):
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
                output = self.merge_list(spec.output_layer, is_multi_hot=spec.is_multi_hot)
                output_layers.append(output)
            else:  # Numeric layers
                projected_layers.append(spec.output_layer)

        if len(sequence_layers) > 0:
            if self.sequence_processor is None:
                raise ValueError("Sequence processor must be provided when specifying seuqence layers")
            x = self.sequence_processor.combine(sequence_layers)
            x = self.sequence_processor.attention(x)
            x = self.merge_list(x, is_multi_hot=True)
            output_layers.append(x)
        if len(projected_layers) > 0:
            if self.numeric_processor is None:
                raise ValueError("Numeric processor must be provided when specifying numeric layers")
            layers = self.numeric_processor.project(projected_layers)
            output_layers.extend(layers)
        output = tf.keras.layers.Concatenate(axis=-1)(output_layers)
        return input_layers, output

    def merge_list(self, x, is_multi_hot=False):
        return get_combiner(self.combiner, is_multi_hot)(x)

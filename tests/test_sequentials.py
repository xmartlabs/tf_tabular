from tensorflow.keras import Model
from tf_tabular.builder import InputBuilder
from tf_tabular.sequence_processor import SequenceProcessor


def test_add_sequential_columns():
    builder = InputBuilder(sequence_processor=SequenceProcessor(attention_name="test_attn"))
    builder.add_inputs_list(
        categoricals=["a", "b"],
        embedding_dims={"a": 10, "b": 20},
        vocabs={"a": [1, 2, 3], "b": [4, 5, 6]},
        sequentials=["a"],
    )
    inputs, output = builder.build_input_layers()
    model = Model(inputs=inputs, outputs=output)
    assert model.get_layer("test_attn") is not None


def test_add_multihot_combiner_default():
    builder = InputBuilder()
    builder.add_inputs_list(categoricals=["a"], embedding_dims={"a": 10}, vocabs={"a": [1, 2, 3]}, multi_hots=["a"])
    inputs, output = builder.build_input_layers()
    model = Model(inputs=inputs, outputs=output)
    assert model.get_layer("a_emb").output_shape == (None, None, 10)
    assert model.get_layer("a_emb").trainable
    assert output.shape == (None, 10)
    assert model.get_layer("global_average_pooling1d_1") is not None


def test_add_multihot_combiner_max():
    builder = InputBuilder(combiner="max")
    builder.add_inputs_list(categoricals=["a"], embedding_dims={"a": 10}, vocabs={"a": [1, 2, 3]}, multi_hots=["a"])
    inputs, output = builder.build_input_layers()
    model = Model(inputs=inputs, outputs=output)
    assert model.get_layer("a_emb").output_shape == (None, None, 10)
    assert output.shape == (None, 10)
    assert model.get_layer("global_max_pooling1d") is not None
    assert model.get_layer("global_max_pooling1d").output_shape == (None, 10)


def test_add_multihot_combiner_sum():
    builder = InputBuilder(combiner="sum")
    builder.add_inputs_list(categoricals=["a"], embedding_dims={"a": 10}, vocabs={"a": [1, 2, 3]}, multi_hots=["a"])
    inputs, output = builder.build_input_layers()
    model = Model(inputs=inputs, outputs=output)
    assert model.get_layer("a_emb").output_shape == (None, None, 10)
    assert output.shape == (None, 10)
    assert model.get_layer("lambda") is not None
    assert model.get_layer("lambda").output_shape == (None, 10)

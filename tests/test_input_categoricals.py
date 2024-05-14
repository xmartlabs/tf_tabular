import pytest
import pandas as pd
import numpy as np
from tensorflow.keras import Model
from tf_tabular.builder import InputBuilder


def test_input_builder_defaults():
    builder = InputBuilder()
    assert builder.input_specs == []
    assert builder.sequence_processor is None
    assert builder.combiner == "mean"
    assert builder.numeric_processor.num_projection is None


def test_add_categoricals_missing_params():
    builder = InputBuilder()
    pytest.raises(KeyError, builder.add_inputs_list, categoricals=["a", "b"])
    pytest.raises(KeyError, builder.add_inputs_list, categoricals=["a", "b"], vocabs={"a": [], "b": []})


def test_add_categoricals_with_embedding():
    builder = InputBuilder()
    builder.add_inputs_list(
        categoricals=["a", "b"], embedding_dims={"a": 10, "b": 20}, vocabs={"a": [1, 2, 3], "b": [4, 5, 6]}
    )
    assert len(builder.input_specs) == 2
    assert builder.input_specs[0].name == "a"
    assert builder.input_specs[1].name == "b"
    assert builder.input_specs[0].embedding_dim == 10
    assert builder.input_specs[1].embedding_dim == 20
    assert not builder.input_specs[0].is_sequence
    assert not builder.input_specs[1].is_sequence
    assert not builder.input_specs[0].is_multi_hot
    assert not builder.input_specs[1].is_multi_hot
    assert builder.input_specs[0].vocab == [1, 2, 3]
    assert builder.input_specs[1].vocab == [4, 5, 6]


def test_add_categoricals_with_embedding_df():
    builder = InputBuilder()
    emb_a = pd.DataFrame({"id": [1, 2, 3], "embedding": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]})
    emb_a["embedding"] = emb_a["embedding"].apply(np.array)
    builder.add_inputs_list(
        categoricals=["a"], embedding_dims={"a": 10}, vocabs={"a": [1, 2, 3]}, embedding_df={"a": emb_a}
    )
    inputs, output = builder.build_input_layers()
    model = Model(inputs=inputs, outputs=output)
    emb_layer = model.get_layer("a_emb")
    assert not emb_layer.trainable

    layer_embs = emb_layer.get_weights()[0]
    expected = np.stack(emb_a.embedding.values).astype(np.float32)

    assert layer_embs.shape == (4, 3)

    assert np.array_equal(layer_embs[1:], expected)
    # assert that the OOV embedding is the mean of the others
    assert np.allclose(layer_embs[0], emb_a.embedding.mean())

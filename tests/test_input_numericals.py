import pytest
import numpy as np
from tensorflow.keras import Model
from tf_tabular.builder import InputBuilder


def test_add_numericals_with_normalization():
    builder = InputBuilder()
    params = {"a": {"sample": np.array([10, 4, 12])}, "b": {"mean": 3.1, "var": 1.0}}
    builder.add_inputs_list(numericals=["a", "b"], normalization_params=params)
    assert len(builder.input_specs) == 2
    assert builder.input_specs[0].name == "a"
    assert builder.input_specs[1].name == "b"
    assert not builder.input_specs[0].is_sequence
    assert not builder.input_specs[1].is_sequence
    assert np.array_equal(builder.input_specs[0].sample, params["a"]["sample"])
    assert builder.input_specs[1].mean == params["b"]["mean"]
    assert builder.input_specs[1].variance == params["b"]["var"]

    inputs, output = builder.build_input_layers()
    model = Model(inputs=inputs, outputs=output)
    assert model.get_layer("a_norm") is not None
    assert model.get_layer("b_norm") is not None


def test_add_numericals_no_norm():
    builder = InputBuilder()
    builder.add_inputs_list(numericals=["a"])
    assert len(builder.input_specs) == 1
    assert builder.input_specs[0].name == "a"
    assert builder.input_specs[0].sample is None
    assert builder.input_specs[0].mean is None
    assert builder.input_specs[0].variance is None

    inputs, output = builder.build_input_layers()
    model = Model(inputs=inputs, outputs=output)
    pytest.raises(ValueError, model.get_layer, "a_norm")

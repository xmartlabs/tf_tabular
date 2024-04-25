# TF Tabular

TF Tabular is a project aimed at simplifying the process of handling tabular data in TensorFlow. It provides utilities for building models on top of numeric, categorical, multihot, and sequential data types.

## Features

- Create input layers based on lists of columns
- Support custom embeddings
- Support attention for mixing sequence layers
- Support multi-hot categoricals
- No model building or training: Build whatever you want on top


## Installation

To get started with TF Tabular, you will need to install it using pip:

```sh
pip install tf-tabular
```

## Usage

Here is a basic example of how to use TF Tabular:

```python
from tf_tabular.builder import InputBuilder

# Define columns to use and specify additional parameters:
categoricals = ['Pclass', 'no_cabin']
numericals = ['Age', 'Fare']
# ....

# Build model:
input_builder = InputBuilder()
input_builder.add_inputs_list(categoricals=categoricals,
                              numericals=numericals,
                              normalization_params=norm_params,
                              vocabs=vocabs,
                              embedding_dims=embedding_dims)
inputs, output = input_builder.build_input_layers()
output = Dense(1, activation='sigmoid')(output)

model = Model(inputs=inputs, outputs=output)
```

<!-- Which will produce a model like this: -->
<!-- TODO: <Insert NETRON view of model> -->

Look at the examples folder for more complete examples.

## Contributing
Contributions to TF Tabular are welcome. If you have a feature you'd like to add, or a bug you'd like to fix, please open a pull request.

## Roadmap:
This is a list of possible features to be added in the future depending on need and interest expressed by the community.

- [ ] Parse dataset to separate numeric vs categoricals, multihots and sequencials
- [ ] Implement other types of normalization
- [ ] Support computing vocab and normalization params?
- [ ] Improve documentation and provide more usage examples

## License
TF Tabular is licensed under the MIT License. See the LICENSE file for more details.

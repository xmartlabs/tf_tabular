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
from tf_tabular.numeric_processor import NumericProcessor

# Define columns to use and specify additional parameters:
categoricals = ['Pclass', 'Sex']
numericals = ['Age', 'Fare']
# ....

# Build model:
input_builder = InputBuilder(numeric_processor=NumericProcessor(num_projection=8, cross_features=False))
input_builder.add_inputs_list(categoricals=categoricals,
                              numericals=numericals,
                              normalization_params=norm_params,
                              vocabs=vocabs,
                              embedding_dims=embedding_dims)
inputs, output = input_builder.build_input_layers()
output = Dense(1, activation='sigmoid')(output)

model = Model(inputs=inputs, outputs=output)
```

Which will produce a model like this:
![Netron Model View](/media/images/example_netron.png)


Look at the examples folder for more complete examples.


## Contributing
Contributions to TF Tabular are welcome. Check out the [contributing](https://github.com/xmartlabs/tf_tabular/CONTRIBUTING.md) guidelines for more details.

### Setting up local development environment
To set up a local development environment, you will need to first clone the repo and then install the required dependencies:
1. Install Poetry follow the instructions on the [official Poetry website](https://python-poetry.org/docs/#installation).
2. Run `poetry install`
3. Run `poetry run pre-commit install` to install git pre-commit

## Roadmap:
This is a list of possible features to be added in the future depending on need and interest expressed by the community.

- [ ] Parse dataset to separate numeric vs categoricals, multihots and sequencials
- [ ] Implement other types of normalization
- [ ] Support computing vocab and normalization params instead of receiving them as parameters
- [ ] Improve documentation and provide more usage examples

## License
TF Tabular is licensed under the MIT License. See the LICENSE file for more details.

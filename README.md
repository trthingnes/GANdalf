# GANdalf
Final project in IDATT2502 Applied Machine Learning at the NTNU.

## Installation
This project uses `pipenv` for managing dependencies.
To install dependencies locally use the command `pipenv`.
To add a new/remove project dependency use `pipenv (un)install`.
To run a command in the virtual environment use `pipenv run <cmd>`.
To open a shell to run commands in use `pipenv shell`.

## Usage
There are several variants of GANs in this project. Some take image labels into consideration like cGAN and cDCGAN, while the rest just generate random images that could be in the dataset. To train a model run the `[model]_train.py` file. If you want to continue from a previous file, use the `--timestamp` option. To sample a model run the `model_sample.py` file (`--timestamp` is required for this).

## Development
The model files are located in a folder with the name of the model type. The util folder contains functions used by all the models, like getting available devices and loading/saving state.

### Running the linter
This project uses `black` to format and lint source code and `isort` to sort imports.
Pull request pipelines will fail if code is not formatted correctly.
To run `black` use `pipenv run black .`.
To run `isort` use `pipenv run isort .`.

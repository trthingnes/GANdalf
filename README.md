# GANdalf
Final project in IDATT2502 Applied Machine Learning at the NTNU.

## Installation
This project uses `pipenv` for managing dependencies.
To install dependencies locally use the command `pipenv`.
To add a new/remove project dependency use `pipenv (un)install`
To run a command in the virtual environment use `pipenv run <cmd>`

## Development
To run the `main.py` file, type `pipenv run python3 src/main.py` while in the project directory.

### Running the linter
This project uses `black` to format and lint source code.
Pull request pipelines will fail if code is not formatted correctly.
To run `black` use `pipenv run black .`

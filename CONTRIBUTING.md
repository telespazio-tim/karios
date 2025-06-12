# CONTRIBUTION GUIDE

## Project structure

The project structure is based on Kenneth Reitz recommendation.

https://docs.python-guide.org/writing/structure/#structure-of-the-repository

## Development environment

Create conda env and install **for development** as explain in [README.md](README.md#environment-setup)

### Additional dependencies

```bash
conda activate karios
conda install -n karios -c conda-forge --file requirements_dev.txt
pip install build # to build wheel
```

It installs: 
- `bandit`: Bandit is a tool for finding common security issues in Python code 
- `isort` : sort import
- `black`: format code
- `pylint`: code linter
- `pytest-cov`: pytest with code coverage support
- `pycodestyle`: check docstring
- `pre-commit`: pre commit hook engine : https://pre-commit.com/
- `setuptools` : for packaging

### Deploy pre-commit hook

pre-commit creates and install git pre commit hook configured thanks to [`.pre-commit-config.yaml`](.pre-commit-config.yaml)

You **MUST** deploy pre-commit !

```bash
pre-commit install && pre-commit install --hook-type pre-push
```

Try it

```bash
pre-commit run
```

## Best Practices

### Format the code

**Please, format the code before commit** by using the following command :

```bash
isort --profile black . && black -l 100 .
```

As an alternative, you can configure your IDE to let it do it for you.

**Use linter to check code quality**:

```bash
pylint karios/
```

### Run tests

**Test Data**

- Retrieve test data [here](**TODO**), 
- Create a folder `/data/KARIOS/` 
- Extract test data archive in the folder `/data/KARIOS/`

**Run tests before push**:

```bash
pytest -s --cov=karios --cov-report html:cov_html tests; \
    python -c 'import webbrowser; webbrowser.open_new_tab("./cov_html/index.html")'
```

For reports options take a look at this : https://pytest-cov.readthedocs.io/en/latest/reporting.html


## Architecture

KARIOS tries follows clean architecture principles with clear separation of concerns:

### Core Components

- **Configuration Layer**: Manages processing parameters and runtime settings
- **Core Layer**: Image handling, geometric operations, error definitions
- **Matcher Layer**: KLT feature tracking and large offset detection
- **Analysis Layer**: Statistical accuracy computation
- **Report Layer**: Visualization and product generation
- **API Layer**: Clean interface for library usage
- **CLI Layer**: Command-line interface

### Design Principles

- **Separation of Concerns**: Configuration defines "how to process", input parameters define "what to process"
- **Dependency Injection**: Components receive their dependencies rather than creating them
- **Reusability**: Same configuration can process multiple image pairs
- **Type Safety**: Comprehensive type hints throughout the codebase
- **Clean Interfaces**: Clear dataclasses for inputs and outputs

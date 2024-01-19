# CONTRIBUTION GUIDE

## Project structure

The project structure is based on Kenneth Reitz recommendation.

https://docs.python-guide.org/writing/structure/#structure-of-the-repository

## Development environment

Create conda env as explain in [README.md](README.md#prerequisite)

### Additional dependencies

```bash
conda install -n karios -c conda-forge --file requirements_dev.txt
```

It installs: 
- `bandit`: Bandit is a tool for finding common security issues in Python code 
- `isort` : sort import
- `black`: format code
- `pylint`: code linter
- `pytest-cov`: pytest with code coverage support
- `pycodestyle`: check docstring
- `pre-commit`: pre commit hook engine : https://pre-commit.com/

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
PYTHONPATH=karios pylint karios/
```

### Run tests

**Test Data**

- Retrieve test data [here](**TODO**), 
- Create a folder `/data/KARIOS/` 
- Extract test data archive in the folder `/data/KARIOS/`

**Run tests before push**:

```bash
PYTHONPATH=karios \
    pytest -s --cov=karios --cov-report html:cov_html tests; \
    python -c 'import webbrowser; webbrowser.open_new_tab("./cov_html/index.html")'
```

For reports options take a look at this : https://pytest-cov.readthedocs.io/en/latest/reporting.html

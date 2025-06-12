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

## Contribute to KARIOS website

See [dedicated Readme of the website](site/README.md)

## Release Process

### Overview

This project uses an automated release workflow that creates draft releases when version tags are pushed to the main branch. The workflow builds the Python wheel, packages it with documentation, and creates a GitHub release ready for review and publication.

### Release Workflow Steps

#### 1. Prepare for Release

Before creating a release:
- Ensure all changes are merged to the `main` branch
- Update version numbers in your code (if not using dynamic versioning)
- Test the build locally to ensure everything works correctly

#### 2. Create and Push a Version Tag

Create a version tag following semantic versioning (e.g., `v1.0.0`, `v1.2.3`, `v2.0.0-beta.1`):

```bash
# Create a new tag
git tag v1.0.0

# Push the tag to trigger the release workflow
git push origin v1.0.0
```

**Important**: Only tags starting with `v` will trigger the release workflow.

#### 3. Automated Build Process

When you push a version tag, the GitHub Actions workflow automatically:

1. **Builds the Python wheel** from your `pyproject.toml` configuration
2. **Creates a release package** containing:
   - The Python wheel file (`.whl`)
   - README documentation (`README.md`)
   - Environment configuration (`environment.yml`)
3. **Packages everything** into a ZIP file named `karios-{version}.zip`
4. **Creates a draft GitHub release** with:
   - The ZIP package attached
   - A basic release description template

#### 4. Review and Publish the Release

After the workflow completes:

1. Navigate to the **Releases** section of the GitHub repository
2. Find the draft release (marked with a "Draft" label)
3. Click **Edit** to review and modify the release:
   - **Update the description** with detailed changelog, breaking changes, new features, etc.
   - **Verify attached files** are correct
   - **Add any additional notes** for users
4. When satisfied, click **Publish release** to make it public

### Release Package Contents

Each release includes:

- **Python Wheel** (`.whl`): Installable package file
- **README.md**: Project documentation and usage instructions
- **environment.yml**: Conda environment configuration
- **Complete ZIP package**: All files bundled together for easy distribution

### Versioning Guidelines

Follow [Semantic Versioning](https://semver.org/) principles:

- **Major version** (`v2.0.0`): Breaking changes, incompatible API changes
- **Minor version** (`v1.1.0`): New features, backwards compatible
- **Patch version** (`v1.0.1`): Bug fixes, backwards compatible
- **Pre-release** (`v1.0.0-beta.1`): Testing versions, not for production

### Troubleshooting

#### Release Workflow Fails
- Check that your `pyproject.toml` is valid
- Ensure all dependencies are properly specified
- Verify the tag follows the `v*` pattern

#### Draft Release Not Created
- Confirm the tag was pushed to the `main` branch
- Check the Actions tab for workflow execution status
- Verify you have the necessary repository permissions

### Missing Files in Release
- Ensure `README.md` exists in the repository root
- Check that `environment.yml` exists in the repository root
- Verify the build process completes successfully

### Best Practices

- **Test locally** before creating releases
- **Use descriptive commit messages** leading up to releases
- **Document breaking changes** clearly in release notes
- **Keep release notes user-focused** rather than technical
- **Consider pre-releases** for major changes to gather feedback

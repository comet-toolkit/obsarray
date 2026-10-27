# obsarray: Measurement uncertainty handling in Python

[![Tests](https://github.com/comet-toolkit/obsarray/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/comet-toolkit/obsarray/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/comet-toolkit/obsarray/branch/main/graph/badge.svg?token=PZTGG03VQY)](https://codecov.io/gh/comet-toolkit/obsarray)
[![Documentation Status](https://readthedocs.org/projects/obsarray/badge/?version=latest)](https://obsarray.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/obsarray)](https://pypi.org/project/obsarray/)

**obsarray** is an extension to [xarray](https://docs.xarray.dev/en/stable/) for defining, storing and interfacing with uncertainty information using standardised metadata. It is particularly designed to work well with [netCDF](https://www.unidata.ucar.edu/software/netcdf/) files and for the Earth Observation community.

obsarray is part of the [CoMet Toolkit](https://www.comet-toolkit.org) (community metrology toolkit), and can combined with the [punpy](https://punpy.readthedocs.io/en/latest/) (propagating uncertainties in python) module for very simple propagation of defined data uncertainties through arbitrary python functions.

## Installation

obsarray is installable via pip:

```bash
pip install obsarray
```

## Documentation

For more information visit our [documentation](https://obsarray.readthedocs.io/en/latest).

## Development

Clone the repository and install it in editable mode with the development
extras:

```bash
pip install -e ".[dev]"   # Development tools (ruff, mypy, pytest, …)
pip install -e ".[docs]"  # Documentation build (sphinx, …)
```

Install the pre-commit hooks after cloning:

```bash
pre-commit install
```

When you commit, `ruff` will lint and format your code and `mypy` will
type-check it. If it makes corrections the commit will be aborted so you can
review the changes — just commit again once you are happy.

Run the test suite with:

```bash
pytest
```

### Releasing

The package version is derived from git tags by
[setuptools_scm](https://setuptools-scm.readthedocs.io), so there is no
version number to edit in the source. To release, tag `main` and push the tag:

```bash
git tag v<X.Y.Z>
git push origin v<X.Y.Z>
```

This triggers the `publish.yml` GitHub Actions workflow, which runs the tests,
builds the sdist and wheel, and publishes them to PyPI via trusted publishing.
The workflow can also be run manually from the GitHub Actions UI (type
`publish` in the confirmation input).

## Compatibility

obsarray requires Python 3.11 or later and is tested on Python 3.11, 3.12,
and 3.13.

## License

obsarray is free software licensed under the
[GNU Lesser General Public License (v3)](./LICENSE).

## Acknowledgements

obsarray has been developed by the [CoMet Toolkit Team](mailto:team@comet-toolkit.org).

The development has been funded by:

* The UK's Department for Business, Energy and Industrial Strategy's (BEIS) National Measurement System (NMS) programme
* The IDEAS-QA4EO project funded by the European Space Agency.

## Project status

obsarray is under active development. It is beta software.

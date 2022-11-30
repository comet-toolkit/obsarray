# obsarray: Measurement uncertainty handling in Python

[![Build Status](https://app.travis-ci.com/comet-toolkit/obsarray.svg?branch=main)](https://app.travis-ci.com/comet-toolkit/obsarray)
[![codecov](https://codecov.io/gh/comet-toolkit/obsarray/branch/main/graph/badge.svg?token=PZTGG03VQY)](https://codecov.io/gh/comet-toolkit/obsarray)
[![Documentation Status](https://readthedocs.org/projects/obsarray/badge/?version=latest)](https://obsarray.readthedocs.io/en/latest/?badge=latest)

**obsarray** is an extension to [xarray](https://docs.xarray.dev/en/stable/) for defining, storing and interfacing with uncertainty information using standardised metadata. It is particularly designed to work well with [netCDF](https://www.unidata.ucar.edu/software/netcdf/) files and for the Earth Observation community.

obsarray is part of the [CoMet Toolkit](https://www.comet-toolkit.org) (community metrology toolkit), and can combined with the [punpy](https://punpy.readthedocs.io/en/latest/) (propagating uncertainties in python) module for very simple propagation of defined data uncertainties through arbitrary python functions.

## Installation

obsarray is installable via pip.

## Documentation

For more information visit our [documentation](https://obsarray.readthedocs.io/en/latest).

## License

obsarray is free software licensed under the
[GNU Public License (v3)](./LICENSE).

## Acknowledgements

obsarray has been developed by [Sam Hunt](https://github.com/shunt16).

The development has been funded by:

* The UK's Department for Business, Energy and Industrial Strategy's (BEIS) National Measurement System (NMS) programme
* The IDEAS-QA4EO project funded by the European Space Agency.

## Project status

obsarray is under active development. It is beta software.
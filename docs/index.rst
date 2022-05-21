obsarray: Measurement uncertainty handling in Python
====================================================

**obsarray** is an extension to `xarray <https://docs.xarray.dev/en/stable/>`_ for defining, storing and interfacing with uncertainty information using standardised metadata. It is particularly designed to work well with `netCDF <https://www.unidata.ucar.edu/software/netcdf/>`_ files and for the Earth Observation community.

obsarray is part of the `CoMet Toolkit <https://www.comet-toolkit.org>`_ (community metrology toolkit), and can combined with the  `punpy <https://punpy.readthedocs.io/en/latest/>`_ (propagating uncertainties in python) module for very simple propagation of defined data uncertainties through arbitrary python functions.

This documentation is under active development.

Acknowledgements
----------------

obsarray has been developed by `Sam Hunt <https://github.com/shunt16>`_.

The development has been funded by:

* The UK's Department for Business, Energy and Industrial Strategy's (BEIS) National Measurement System (NMS) programme
* The IDEAS-QA4EO project funded by the European Space Agency.

Project status
--------------

obsarray is under active development. It is beta software.

.. toctree::
    :hidden:
    :caption: User Guide

    content/user/unc_accessor
    content/user/templater

.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   content/api/obsarray

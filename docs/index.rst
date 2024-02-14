obsarray: Measurement data handling in Python
=============================================

**obsarray** is an extension to `xarray <https://docs.xarray.dev/en/stable/>`_ to support defining, storing and interfacing with measurement data - in particular, measurement uncertainty information using standardised metadata. It is designed to work well with `netCDF <https://www.unidata.ucar.edu/software/netcdf/>`_ files and for the Earth Observation community.

**obsarray** is part of the `CoMet Toolkit <https://www.comet-toolkit.org>`_ (community metrology toolkit), and can combined with the  `punpy <https://punpy.readthedocs.io/en/latest/>`_ (propagating uncertainties in python) module for very simple propagation of defined data uncertainties through arbitrary python functions.

.. grid:: 2
    :gutter: 2

    .. grid-item-card::  Quickstart Guide
        :link: content/user/quickstart
        :link-type: doc

        New to *obsarray*? Check out the quickstart guide for an introduction.

    .. grid-item-card::  User Guide
        :link: content/user/user_guide
        :link-type: doc

        The user guide provides a documentation and examples how to use **obsarray** to handle measurement data.

    .. grid-item-card::  API Reference
        :link: content/user/api
        :link-type: doc

        The API Reference contains a description the **obsarray** API.

    .. grid-item-card::  Developer Guide
        :link: content/developer/developer_guide
        :link-type: doc

        Guide for contributing to **obsarray** (under development).


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
   :maxdepth: 2
   :hidden:
   :caption: For users

   Quickstart <content/user/quickstart>
   User Guide <content/user/user_guide>
   API Reference <content/user/api>

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: For developers/contributors

   Developer Guide <content/developer/developer_guide>

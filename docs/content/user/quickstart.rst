.. _quickstart:

################
Quickstart Guide
################

Installation
------------

**obsarray** is installable via pip.

.. code-block::

   pip install obsarray


Dependencies
------------

**obsarray** is an extension to `xarray <https://docs.xarray.dev/en/stable/>`_ to support defining, storing and interfacing with measurement data. It is designed to work well with `netCDF <https://www.unidata.ucar.edu/software/netcdf/>`_ files, using the `netcdf4 <https://unidata.github.io/netcdf4-python/>`_ library.

The pip installation will also automatically install any dependencies.


Example Usage
-------------

First we build an example dataset that represents a time series of temperatures (for more on how do this see the `xarray <https://docs.xarray.dev/en/stable/>`_ documentation).

.. ipython:: python

   import numpy as np
   import xarray as xr
   import obsarray

   # build an xarray to represents a time series of temperatures
   temps = np.array([20.2, 21.1, 20.8])
   times = np.array([0, 30, 60])
   ds = xr.Dataset(
      {"temperature": (["time"], temps, {"units": "degC"})},
      coords = {"time": (["time"], times, {"units": "s"})}
   )

Uncertainty and error-covariance information for observation variables can be defined using the dataset's ``unc`` accessor, which is provided by **obsarray**.

.. ipython:: python

   # add random component uncertainty
   ds.unc["temperature"]["u_r_temperature"] = (
      ["time"],
      np.array([0.5, 0.5, 0.6]),
      {"err_corr": [{"dim": "time", "form": "random"}]}
   )
   # add systematic component uncertainty
   ds.unc["temperature"]["u_s_temperature"] = (
      ["time"],
      np.array([0.3, 0.3, 0.3]),
      {"err_corr": [{"dim": "time", "form": "systematic"}]}
   )

Dataset structures can be defined separately using **obsarray**'s :ref:`templating <template>` functionality. This is helpful for processing chains where you want to write files to a defined format.

The defined uncertainty information then can be interfaced with, for example:

.. ipython:: python

   # get total combined uncertainty of all components
   ds.unc["temperature"].total_unc()
   # get total error-covariance matrix for all components
   ds.unc["temperature"].total_err_cov_matrix()

This information is preserved in metadata when written to netCDF files

.. ipython:: python

   # show uncertainty components
   ds.unc["temperature"]
   # write file
   ds.to_netcdf("~/temp_ds.nc")
   # reopen file
   ds = xr.open_dataset("~/temp_ds.nc")
   # show uncertainty components
   ds.unc["temperature"]

Similarly, data flags can be defined using the dataset’s ``flag`` accessor, which again is provided by **obsarray**. These flags are defined following the `CF Convention <https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#flags>`_ metadata standard.

A flag variable can be created to store data for a set of flags with defined meanings

.. ipython:: python

    ds.flag["quality_flags"] = (
        ["time"],
        {"flag_meanings": ["dubious", "invalid", "saturated"]}
    )
    print(ds.flag)

These flag meanings can be indexed, to get and set their value

.. ipython:: python

    print(ds.flag["quality_flags"]["dubious"].value)
    ds.flag["quality_flags"]["dubious"][0] = True
    print(ds.flag["quality_flags"]["dubious"].value)
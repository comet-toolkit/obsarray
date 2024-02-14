.. _flag accessor:

.. ipython:: python
   :suppress:

   import numpy as np
   import pandas as pd
   import xarray as xr
   import obsarray

   np.random.seed(123457)
   np.set_printoptions(threshold=6)

   temp = 15 + 8 * np.random.randn(2, 2, 3)
   precip = 10 * np.random.rand(2, 2, 3)
   lon = [[-99.83, -99.32], [-99.79, -99.23]]
   lat = [[42.25, 42.21], [42.63, 42.59]]

   ds = xr.Dataset(
       {
           "temperature": (["x", "y", "time"], temp),
           "precipitation": (["x", "y", "time"], precip),
       },
       coords={
           "lon": (["x", "y"], lon),
           "lat": (["x", "y"], lat),
           "time": pd.date_range("2014-09-06", periods=3),
           "reference_time": pd.Timestamp("2014-09-05"),
       },
   )

   ds.flag["time_flags"] = (["time"], {"flag_meanings": ["dubious", "invalid"]})


=================================
Interfacing with Flag Information
=================================

**obsarray** enables users to define, store and interface with flag variables in :py:class:`xarray.Dataset`'s following the `CF Convention <https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#flags>`_ metadata standard.

To access the information from a particular flag variable, use the dataset's ``flag`` accessor.

You can see which dataset variables are flag variables by looking at the ``flag`` accessor keys.

.. ipython:: python

   import obsarray

   ds.flag.keys()

This means this dataset contains one flag variable called ``"time_flags"``.

Interfacing with a Flag Variable
--------------------------------

To inspect the data for a particular flag defined for a flag variable, index the ``flag`` accessor with it's name.

.. ipython:: python

    ds.flag["time_flags"]


This returns a :py:class:`~obsarray.flag_accessor.FlagVariable` object, which provides an interface to the flags defined by a flag variable. You can see which flags are defined by the flag variable by looking at the its keys.

.. ipython:: python

    ds.flag["time_flags"].keys()

This means the ``time_flags`` variable defines two flags called ``"dubious"`` and ``"invalid"``. Therefore, the first two datum bits for each element of the `time_flags` variable array corresponds to boolean condition flags with these meanings (as per the `CF Convention <https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#flags>`_). **obarray** makes the bit handling for these flags simple.


Interfacing with Flags in Flag Variables
----------------------------------------

To inspect a specific flag of particular flag variable, index the flag variable with its name.


.. ipython:: python

    ds.flag["time_flags"]["dubious"]

This returns a :py:class:`~obsarray.flag_accessor.Flag` object, which provides an interface to a specific uncertainty variable.

The mask that represents the flag can be returned as an :py:class:`xarray.DataArray` as:

.. ipython:: python

   print(ds.flag["time_flags"]["dubious"].value)

Flag values can be set:

.. ipython:: python

   ds.flag["time_flags"]["dubious"][0] = True
   print(ds.flag["time_flags"]["dubious"])

Adding/Removing Flags
---------------------

The same interface can be used to add/remove flags from the dataset. A new flag variable can be added following a similar syntax to the xarray convention, as :python:`ds.flag["flag_var"] = (dims, attributes)`. The attributes must contain a list of ``"flag_meanings"``.

.. ipython:: python

    ds.flag["spatial_flags"] = (
        ["lat", "lon"],
        {"flag_meanings": ["land", "ocean"]}
    )
    print(ds.flag)

A new flag to an existing flag variable as follows,

.. ipython:: python

   ds.flag["spatial_flags"]["ice"] = False
   print(ds.flag)

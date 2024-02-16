.. _unc accessor:

====================================
Interfacing with Dataset Uncertainty
====================================

.. ipython:: python
   :suppress:
   :okwarning:

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

   ds.unc["temperature"]["u_ran_temperature"] = (["x", "y", "time"], temp ** 0.5, {})

   err_corr = [
       {
           "dim": ["x", "y", "time"],
           "form": "systematic",
           "params": [],
           "units": []
       }
   ]
   ds.unc["temperature"]["u_sys_temperature"] = (["x", "y", "time"], np.full(temp.shape, 5.0), {"err_corr": err_corr})
   ds.unc["precipitation"]["u_ran_precipitation"] = (["x", "y", "time"], precip ** 0.5, {})


**obsarray** enables users to define, store and interface with uncertainty information in :py:class:`xarray.Dataset`'s.

To access the uncertainty information for a particular measured variable, use the dataset's ``unc`` accessor.

You can see which dataset variables have uncertainty variables associated with them by looking at ``unc`` accessor keys.

.. ipython:: python
   :okwarning:

   import obsarray

   ds.unc.keys()

This means that the dataset variables ``temperature`` and ``precipitation`` have uncertainty data defined for them. We refer to these variables as "observation variables".

Interfacing with Variable Uncertainty
-------------------------------------

To inspect the uncertainty data defined for a particular observation variable, index the ``unc`` accessor with it's name.

.. ipython:: python
   :okwarning:

   ds.unc["temperature"]


This returns a :py:class:`~obsarray.unc_accessor.VariableUncertainty` object, which provides an interface to an observation variable's uncertainty information. This shows that ``temperature`` has two uncertainty variables - ``u_ran_temperature`` and ``u_sys_temperature``.

To evaluate the total uncertainty for the observation variable, run:

.. ipython:: python
   :okwarning:

   ds.unc["temperature"].total_unc()

which combines all the uncertainty components by sum of squares. Similarly, you can see the combined random or systematic uncertainty components (where more than one component of either is defined), as follows,

.. ipython:: python
   :okwarning:

   ds.unc["temperature"].random_unc()
   ds.unc["temperature"].systematic_unc()

You can also see the combined error-correlation matrix,

.. ipython:: python
   :okwarning:

   ds.unc["temperature"].total_err_corr_matrix()

This gives the cross-element error-correlation between each element in the ``temperature`` array. The order of the observation elements along both dimensions of the error-correlation matrix is defined by the order :py:meth:`flatten()` method produces.

Similar, the error-covariance matrix,

.. ipython:: python
   :okwarning:

   ds.unc["temperature"].total_err_cov_matrix()

You can also do this to access a subset of the total error-covariance matrix by indexing with the slice of interest (this can avoid building the whole error-covariance matrix in memory).

.. ipython:: python
   :okwarning:

   # error-covariance matrix for measurements at one time step
   ds.unc["temperature"][:,:,1].total_err_cov_matrix()


Interfacing with Uncertainty Components
---------------------------------------

To inspect a specific uncertainty component of an observation variable, index the variable uncertainty with its name.

.. ipython:: python
   :okwarning:

    ds.unc["temperature"]["u_ran_temperature"]

This returns a :py:class:`~obsarray.unc_accessor.Uncertainty` object, which provides an interface to a specific uncertainty variable.

The error correlation structure of the uncertainty variable can be inspected as follows:

.. ipython:: python
   :okwarning:

   ds.unc["temperature"]["u_ran_temperature"].err_corr
   ds.unc["temperature"]["u_ran_temperature"].err_corr_dict()
   ds.unc["temperature"]["u_ran_temperature"].err_corr_matrix()

Adding/Removing Uncertainty Components
--------------------------------------

The same interface can be used to add/remove uncertainty components from the dataset, safely handling the uncertainty metadata. This is achieved following a similar syntax to the xarray convention, as :python:`ds.unc["var"]["u_var"] = (dims, values, attributes)`.

To define the error-correlation structure, the attributes must contain an entry called ``err_corr`` with a list that defines the error-correlation structure per data dimension (if omitted the error-correlation is assumed random). How to define these is defined in detail in the dataset templating :ref:`section <err corr>`. See below for an example:

.. ipython:: python
   :okwarning:

   # Define error-correlation structure
   err_corr_def = [
       {
           "dim": ["x", "y"],
           "form": "systematic",
           "params": [],
           "units": []
       },
       {
           "dim": ["time"],
           "form": "random",
           "params": [],
           "units": []
       }
   ]

   # Define uncertainty values at 5%
   unc_values = ds["temperature"] * 0.05

   # Add uncertainty variable
   ds.unc["temperature"]["u_str_temperature"] = (
       ["x", "y", "time"],
       unc_values,
       {"err_corr": err_corr_def, "pdf_shape": "gaussian"}
   )

   # Check uncertainties
   ds.unc["temperature"].keys()

A component of uncertainty can be simply be deleted as,

.. ipython:: python
   :okwarning:

   del ds.unc["temperature"]["u_str_temperature"]

   # Check uncertainties
   ds.unc["temperature"].keys()

Renaming Variables
------------------

The storage of uncertainty information is underpinned by variable attributes, which include referencing other variables (for example, which variables are the uncertainties associated with a particular observation variable). Because of this it is important, if renaming uncertainty variables, to use **obsarray**'s renaming functionality. This renames the uncertainty variable and safely updates attribute variable references. This is done as follows:


.. ipython:: python
   :okwarning:

   print(ds.unc["temperature"])
   ds = ds.unc["temperature"]["u_ran_temperature"].rename("u_noise")
   print(ds.unc["temperature"])
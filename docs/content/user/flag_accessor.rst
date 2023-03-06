Interfacing with Flag Information
+++++++++++++++++++++++++++++++++

obsarray enables users to define, store and interface with flag variables in :py:class:`xarray.Dataset`'s following the `CF Convention <https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#flags>`_ metadata standard.

To access the information from a particular flag variable, use the dataset's ``flag`` accessor as follows,

.. code-block:: python

    import xarray as xr
    import obsarray

    ds = xr.open_dataset("measurement_data.nc")
    print(ds.flag["flag_var"])

This lists the flags for that variable. To access the flag values for a particular flag variable,

.. code-block:: python

    ds.flag["flag_var"]["flag_1"].value         # get flag mask
    ds.flag["flag_var"]["flag_1"][:, 0] = True  # set flag mask values



You can add a flag variable in a similar way to adding a new variable to a dataset,

.. code-block:: python

    ds.flag["new_flag_var"] = (["dims"], {"flag_meanings": ["f1", "f2"]})

or add a new flag to an existing flag variable,

.. code-block:: python

    ds.flag["new_flag_var"]["f3"] = False

Interfacing with Uncertainty Information
++++++++++++++++++++++++++++++++++++++++

obsarray enables users to define, store and interface with uncertainty information in :py:class:`xarray.Dataset`'s.

To access the uncertainty information for a particular measured variable, use the dataset's ``unc`` accessor as follows,

.. code-block:: python

    import xarray as xr
    import obsarray

    ds = xr.open_dataset("measurement_data.nc")
    print(ds.unc["measurand"])

This lists the uncertainty components for that variable. To access the uncertainty information for a particular variable,

.. code-block:: python

    ds.unc["measurand"]["u_rand_measurand"]
    ds.unc["measurand"]["u_rand_measurand"].value     # uncertainty data
    ds.unc["measurand"]["u_rand_measurand"].err_corr  # error correlation information


Or evaluate the combined uncertainty from all the uncertainty components,

.. code-block:: python

    ds.unc["measurand"].total

You can add a new uncertainty component for a particular variable in a similar way to adding a new variable to a dataset,

.. code-block:: python

    unc_meta = {"err_corr": err_corr_dict}

    ds.unc["measurand"]["u_new_measurand"] = (["dims"], unc_data, unc_meta)

Where the ``unc_meta`` dictionary must contain an entry that defines the error-correlation structure between measurement due to this uncertainty component (i.e., random or systematic). How to define this error-correlation metadata dictionary is described in the more detailed :ref:`uncertainty variables <err corr>` section.



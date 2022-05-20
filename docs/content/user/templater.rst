Creating template datasets
++++++++++++++++++++++++++

Dataset definition structures
=============================

**obsarray** can create :py:class:`xarray.Dataset`'s to a particular templates, defined as a :py:class:`dict`'s (referred to hereafter as **template** dictionaries), which can range from very simple to more complex. Every key in the **template** dictionary is the name of a variable, with the corresponding entry a further variable specification dictionary (referred to hereafter as **variable** dictionaries).

So a **template** dictionary may look something like this:

.. code-block:: python

    template = {
        "temperature": temperature_variable,
        "u_temperature": u_temperature_variable
    }

Each **variable** dictionary defines the following entries:

* ``dim`` - list of variable dimension names.
* ``dtype`` - variable data type, generally a :py:class:`numpy.dtype`, though for some :ref:`special variables <special variables>` particular values may be required.
* ``attributes`` - dictionary of variable metadata, for some :ref:`special variables <special variables>` particular entries may be required.
* ``encoding`` - (optional) variable `encoding <http://xarray.pydata.org/en/stable/user-guide/io.html?highlight=encoding#writing-encoded-data>`_.

So for the previous example we may define:

.. _variables spec ex:

.. code-block:: python

    import numpy as np

    temperature_variable = {
        "dim": ["lon", "lat", "time"]
        "dtype": np.float32,
        "attributes": {"units": "K", "unc_comps": ["u_temperature"]}
    }

    u_temperature_variable = {
        "dim": ["lon", "lat", "time"]
        "dtype": np.float16,
        "attributes": {"units": "%"}
    }

The following section details the special variable types that can be defined with **obsarray**.

.. _special variables:

Special variable types
~~~~~~~~~~~~~~~~~~~~~~

**obsarray**'s special variables allow the quick definition of a set of standardised variable formats. The following special variable types are available.

.. _err corr:

Uncertainties
_____________

`Recent work <https://www.mdpi.com/2072-4292/11/5/474/htm>`_ in the Earth Observation metrology domain is working towards the standardisation of the representation of measurement uncertainty information in data, with a particular focus on capturing the error-covariance associated with the uncertainty. Although it is typically the case that for large measurement datasets storing full error-covariance matrices is impractical, often the error-covariance between measurements may be efficiently parameterised. Work to standardise such parameterisations is on-going (see for example the EU H2020 FIDUCEO project defintions list in Appendix A of `this project report <https://ec.europa.eu/research/participants/documents/downloadPublic?documentIds=080166e5c84c9e2c&appId=PPGMS>`_).

**dsbuilder** enables the specification of such error-correlation parameterisations for uncertainty variables through the variable attributes. This is achieved by including an ``"err_corr"`` list entry in a variable's **variable_spec** dictionary. Each element of ``err_corr`` is a  dictionary defining the error-correlation along one or more dimensions, which should include the following entries:

* ``dim`` (*str*/*list*) - name of the dimension(s) as a str or list of str's (i.e. from ``dim_names``)
* ``form`` (*str*) - error-correlation form, defines functional form of error-correlation structure along
  dimension. Suggested error-correlation forms are defined in a :ref:`table below <err corr params table>`.
* ``params`` (*list*) - (optional) parameters of the error-correlation structure defining function for dimension
  if required. The number of parameters required depends on the particular form.
* ``units`` (*list*) - (optional) units of the error-correlation function parameters for dimension
  (ordered as the parameters)

Measurement variables with uncertainties should include a list of ``unc_comps`` in their attributes, as in the :ref:`above example <variables spec ex>`.

An example ``err_corr`` dictionary may therefore look like:

.. code-block:: python

    err_corr = {
        {
            "dim": "x"
            "form": "rectangular_absolute",
            "params": [val1, val2],
            "units": ["m", "m"]
        },
        {
            "dim": "y"
            "form": "random",
            "params": [],
            "units": []
        }
    }


If the error-correlation structure is not defined along a particular dimension (i.e. it is not included in ``err_corr``), the error-correlation is assumed random. Variable attributes are populated to the effect of this assumption.

.. _err corr params table:
.. list-table:: Suggested error correlation parameterisations
   :widths: 25 25 50
   :header-rows: 1

   * - Form Name
     - Parameters
     - Description
   * - ``"random"``
     - None required
     - Errors uncorrelated along dimension(s)
   * - ``"systematic"``
     - None required
     - Errors fully correlated along dimension(s)
   * - ``"custom"``
     - Error-correlation matrix variable name
     - Error-correlation for dimension(s) not parameterised, defined as a full matrix in another named variable in dataset.


.. _flags:

Flags
_____

Setting the ``"flag"`` dtype builds a variable in the `cf conventions flag format <https://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#flags>`_. Each datum bit corresponds to boolean condition flag with a given meaning.

The variable must be defined with an attribute that lists the per bit flag meanings as follows:

.. code-block:: python

   variables = {
       "quality_flag": {
           "dim": ["x", "y"],
           "dtype": "flag"
           "attributes": {
               "flag_meanings": ["good_data", "bad_data"]
           }
       }
   }

The smallest necessary integer is used as the flag variable :py:class:`numpy.dtype`, given the number of flag meanings defined (i.e. 7 flag meanings results in an 8 bit integer variable).

Creating a template dataset
===========================

With the ``template`` dictionary prepared, only two more specifications are required to build a template dataset. First a dictionary that defines the sizes of all the dimensions used in the ``template`` dictionary, e.g.:

.. code-block:: python

   dim_size= {"x": 1000, "y": 2000}


Secondly, a dictionary of dataset global metadata, e.g.:

.. code-block:: python

   metadata = {"dataset_name": "my cool image"}


Combining the above together a template dataset can be created as follows:

.. code-block:: python

   ds = obsarray.create_ds(
       template,
       dim_sizes,
       metadata
   )

Where ``ds`` is an empty xarray dataset with variables defined by the template definition. Fill values for the empty arrays are chosen using the `cf convention values <http://cfconventions.org/cf-conventions/cf-conventions.html#missing-data>`_.

Populating and writing the dataset
----------------------------------

`Populating <http://xarray.pydata.org/en/stable/user-guide/data-structures.html#dictionary-like-methods>`_ and `writing <http://xarray.pydata.org/en/stable/user-guide/io.html#reading-and-writing-files>`_ the dataset can be achieved using xarray's builtin functionality. Here's a dummy example:

.. code-block:: python

   ds["band_red"] = ... # populate variable with red image array
   ds["band_green"] = ... # populate variable with green image array
   ds["band_blue"] = ... # populate variable with blue image array

   ds.to_netcdf("path/to/file.nc")


.. code-block:: python

        import numpy as np

        # define ds variables
        template = {
            "temperature": {
                "dtype": np.float32,
                "dim": ["x", "y", "time"],
                "attrs": {
                    "units": "K",
                    "u_components": ["u_temperature"]
                }
            },
            "u_temperature": {
                "dtype": np.int16,
                "dim": ["x", "y", "time"],
                "attrs": {"units": "%"},
                "err_corr": [
                    {
                        "dim": "x",
                        "form": "systematic",
                        "params": [],
                        "units": []
                    }
                ]
            },
            "quality_flag_time": {
                "dtype": "flag",
                "dim": ["time"],
                "flag_meanings": ["bad", "dubious"]
            },
        }

        # define dim_size_dict to specify size of arrays
        dim_sizes = {
            "x": 500,
            "y": 600,
            "time": 6
        }

        # create dataset
        ds = obsarray.create_ds(template, dim_sizes)

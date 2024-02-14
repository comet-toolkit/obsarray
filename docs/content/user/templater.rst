=================================
Creating Datasets Using Templates
=================================

Basic template definition
-------------------------

**obsarray** can create :py:class:`xarray.Dataset`'s to a particular template, defined by a :py:class:`dict` (referred to hereafter as **template** dictionaries).

Every key in the **template** dictionary is the name of a variable, with the corresponding entry a further variable specification dictionary (referred to hereafter as **variable** dictionaries). Each **variable** dictionary defines the following entries:

* ``dim`` - list of variable dimension names.
* ``dtype`` - variable data type, generally a :py:class:`numpy.dtype`, though for some :ref:`special variables <special variables>` particular values may be required.
* ``attributes`` - dictionary of variable metadata, for some :ref:`special variables <special variables>` particular entries may be required.
* ``encoding`` - (optional) variable `encoding <http://xarray.pydata.org/en/stable/user-guide/io.html?highlight=encoding#writing-encoded-data>`_.

Altogether then, we can define the following basic template:

.. _variables spec ex:

.. ipython:: python

    import numpy as np

    temp_var_dict = {
        "dim": ["lon", "lat", "time"],
        "dtype": np.float32,
        "attributes": {"units": "K"}
    }
    template = {
        "temp": temp_var_dict,
    }

.. _special variables:

Special variable types
----------------------

**obsarray**'s special variables allow the quick definition of variables in a set of standardised templates. The following section describes the types of special variable available and how to define them in a template.

.. _flags:

Flags
~~~~~

Setting the ``dtype`` as ``"flag"`` in the **variable** dictionary builds a variable in the `cf conventions flag format <https://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#flags>`_. Each datum bit corresponds to boolean condition flag with a given meaning.

The variable must be defined with a ``"flag_meanings"`` attribute that lists the per bit flag meanings as follows:

.. ipython:: python

   variables = {
       "quality_flag": {
           "dim": ["x", "y"],
           "dtype": "flag",
           "attributes": {
               "flag_meanings": ["good_data", "bad_data"]
           }
       }
   }

The smallest necessary integer is used as the flag variable :py:class:`numpy.dtype`, given the number of flag meanings defined (i.e., defining 7 flag meanings results in an 8-bit integer variable).

Once built, flag variables can be interfaced with via the **obsarray**'s ``flag`` accessor (extension to :py:class:`xarray.Dataset`) - see the :ref:`section <flag accessor>` on interfacing with flags for more.

.. _err corr:

Uncertainties
~~~~~~~~~~~~~

`Recent work <https://www.mdpi.com/2072-4292/11/5/474/htm>`_ in the Earth Observation metrology domain is working towards the standardisation of the representation of measurement uncertainty information in data, with a particular focus on capturing the error-covariance associated with the uncertainty. Although it can be the case that for large, multi-dimensional arrays of measurements storing a full error-covariance matrix would be impractical, often the error-covariance between measurements may be efficiently parameterised. Work to standardise such parameterisations is on-going (see for example the EU H2020 FIDUCEO project defintions list in Appendix A of `this project report <https://ec.europa.eu/research/participants/documents/downloadPublic?documentIds=080166e5c84c9e2c&appId=PPGMS>`_).

**obsarray** enables the specification of such error-correlation parameterisations for uncertainty variables through the variable attributes. This is achieved by including an ``"err_corr"`` entry in the **variable** dictionary ``attributes``. This ``"err_corr"`` entry is a list of dictionaries defining the error-correlation along one or more dimensions, which should include the following entries:

* ``dim`` (*str*/*list*) - name of the dimension(s) as a str or list of str's (i.e. from ``dim_names``)
* ``form`` (*str*) - error-correlation form, defines functional form of error-correlation structure along
  dimension. Suggested error-correlation forms are defined in a :ref:`table below <err corr params table>`.
* ``params`` (*list*) - (optional) parameters of the error-correlation structure defining function for dimension
  if required. The number of parameters required depends on the particular form.
* ``units`` (*list*) - (optional) units of the error-correlation function parameters for dimension
  (ordered as the parameters)

Measurement variables with uncertainties should include a list of ``unc_comps`` variable names in their attributes.


.. _err corr params table:
.. list-table:: Suggested error correlation parameterisations (to be extended in future)
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


Updating the above example to include an uncertainty component, we can therefore define:

.. ipython:: python

   import numpy as np

   temp_var_dict = {
       "dim": ["lon", "lat", "time"],
       "dtype": np.float32,
       "attributes": {"units": "K"}
   }
   u_temp_var_dict = {
       "dim": ["lon", "lat", "time"],
       "dtype": np.float16,
       "attributes": {
           "units": "K",
           "err_corr": [{"dim": ["lat", "lon"], "form": "systematic"}]
       }
   }
   template = {
       "temp": temp_var_dict,
       "u_temp": u_temp_var_dict,
   }

If the error-correlation structure is not defined along a particular dimension (i.e. it is not included in ``err_corr``), the error-correlation is assumed random in this dimension. So, in the above example, the ``u_temp`` uncertainty is defined to be systematic between all spatial points (i.e., across the ``lat`` and ``lon`` dimensions) at each time step, but random between time steps  (i.e, along the ``time`` dimension) as this is not explicitly defined.

Once built, uncertainty variables can be interfaced with via the **obsarray**'s ``unc`` accessor (extension to :py:class:`xarray.Dataset`) - see the :ref:`section <unc accessor>` on interfacing with data uncertainty for more.

Creating a template dataset
---------------------------

With the ``template`` dictionary prepared, only two more specifications are required to build a template dataset. First a dictionary that defines the sizes of all the dimensions used in the ``template`` dictionary, e.g.:

.. ipython:: python

   dim_sizes = {"lat": 20, "lon": 10, "time": 5}


Secondly, a dictionary of dataset global metadata, e.g.:

.. ipython:: python

   metadata = {"dataset_name": "temperature dataset"}


Combining the above together a template dataset can be created as follows:

.. ipython:: python

   import obsarray
   ds = obsarray.create_ds(
       template,
       dim_sizes,
       metadata
   )
   print(ds)

Where ``ds`` is an empty xarray dataset with variables defined by the template definition. Fill values for the empty arrays are chosen using the `cf convention values <http://cfconventions.org/cf-conventions/cf-conventions.html#missing-data>`_.

`Populating <http://xarray.pydata.org/en/stable/user-guide/data-structures.html#dictionary-like-methods>`_ and `writing <http://xarray.pydata.org/en/stable/user-guide/io.html#reading-and-writing-files>`_ the dataset can then be achieved using xarray's builtin functionality.
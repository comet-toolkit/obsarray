.. _unc_conventions:

#############################################
Upcoming change: UNC specification draft v0.2
#############################################

**obsarray** stores uncertainty information following the `Uncertainty metadata Naming Conventions (UNC)
specification <https://comet-toolkit.github.io/unc_website/>`_. It currently follows
`v0.1 <https://comet-toolkit.github.io/unc_website/specification/v0.1/unc_specification.html>`_.

`Draft v0.2 <https://comet-toolkit.github.io/unc_website/specification/draft-v0.2/unc_specification.html>`_
is a major revision, developed with the CF community and proposed for inclusion in the
`CF conventions <https://cfconventions.org>`_ as a new "Uncertainty" chapter. A future release of **obsarray**
will support draft v0.2 and make it the default. Support for v0.1 will remain available, selectable with the
``unc_conventions`` option (see :ref:`below <unc_conventions_option>`).

What will change
================

Datasets following v0.1 are **not compatible** with draft v0.2. The main changes to the metadata obsarray writes
are:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - v0.1 (current)
     - Draft v0.2
   * - ``unc_comps`` attribute on the observation variable (list of names)
     - ``uncertainty_variables`` attribute on the data variable (blank-separated string of names)
   * - ``pdf_shape`` attribute (default ``"gaussian"``)
     - ``probability_distribution`` attribute
   * - ``err_corr_dim<i>_name``, ``_form``, ``_params`` and ``_units`` attributes
     - Single ``error_correlation`` string attribute
   * - Error-correlation forms ``random`` and ``systematic``
     - Error-correlation structural types ``independent`` and ``common``
   * - Meaning of the uncertainty values not defined
     - Required ``coverage_interval`` and ``coverage_probability`` attributes

The full list is in `Appendix B of the draft v0.2 specification
<https://comet-toolkit.github.io/unc_website/specification/draft-v0.2/unc_specification.html#appendix-b-changes-from-v0-1>`_.

The UNC conventions warning
===========================

Until a conventions version is chosen explicitly, **obsarray** issues an :py:class:`~obsarray.conventions.UNCConventionsWarning`
the first time in a session that it writes uncertainty metadata (for example, when adding an uncertainty variable
with ``ds.unc[...]`` or creating a dataset from a template that includes one). It is shown once per session, and
reading existing uncertainty information does not trigger it.

.. _unc_conventions_option:

Choosing the conventions version
================================

To keep the current behaviour and silence the warning, choose v0.1 explicitly with
:py:class:`~obsarray.options.set_options`, either for the rest of the session:

.. code-block:: python

    import obsarray

    obsarray.set_options(unc_conventions="v0.1")

or only within a block:

.. code-block:: python

    with obsarray.set_options(unc_conventions="v0.1"):
        ds.unc["temperature"]["u_noise"] = (["x"], u_noise, {"units": "K"})

``"v0.1"`` is currently the only supported value. ``"v0.2"`` will become available once draft v0.2 support is
added.

The warning can also be filtered like any other Python warning:

.. code-block:: python

    import warnings
    import obsarray

    warnings.filterwarnings("ignore", category=obsarray.UNCConventionsWarning)

.. note::
    **For developers of packages built on obsarray:** if your package writes uncertainty metadata on behalf of
    its users, wrap that code in ``with obsarray.set_options(unc_conventions="v0.1"):``. This keeps your output
    unchanged when obsarray's default changes, and avoids showing your users a warning about code they don't
    control. Avoid calling ``obsarray.set_options`` globally from library code, as that changes the setting for
    your users' own obsarray code too.

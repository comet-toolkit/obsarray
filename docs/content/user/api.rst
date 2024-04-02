.. currentmodule:: obsarray

.. _api:

#############
API reference
#############

This page provides an auto-generated summary of **obsarray**'s API. For more details
and examples, refer to the relevant chapters in the main part of the
documentation.

Templating functions
====================

.. autosummary::
   :toctree: generated/

   templater.template_util.create_ds
   templater.dstemplater.DSTemplater

Uncertainty functions
=====================

.. autosummary::
   :toctree: generated/

   unc_accessor.UncAccessor
   unc_accessor.UncAccessor.obs_vars
   unc_accessor.UncAccessor.unc_vars
   unc_accessor.UncAccessor.__getitem__
   unc_accessor.UncAccessor.keys
   unc_accessor.VariableUncertainty
   unc_accessor.VariableUncertainty.__getitem__
   unc_accessor.VariableUncertainty.__setitem__
   unc_accessor.VariableUncertainty.__delitem__
   unc_accessor.VariableUncertainty.keys
   unc_accessor.VariableUncertainty.comps
   unc_accessor.VariableUncertainty.random_comps
   unc_accessor.VariableUncertainty.structured_comps
   unc_accessor.VariableUncertainty.systematic_comps
   unc_accessor.VariableUncertainty.total_unc
   unc_accessor.VariableUncertainty.random_unc
   unc_accessor.VariableUncertainty.structured_unc
   unc_accessor.VariableUncertainty.systematic_unc
   unc_accessor.VariableUncertainty.total_err_corr_matrix
   unc_accessor.VariableUncertainty.structured_err_corr_matrix
   unc_accessor.VariableUncertainty.total_err_cov_matrix
   unc_accessor.VariableUncertainty.structured_err_cov_matrix
   unc_accessor.Uncertainty
   unc_accessor.Uncertainty.err_corr_dict
   unc_accessor.Uncertainty.err_corr_matrix
   unc_accessor.Uncertainty.err_cov_matrix
   unc_accessor.Uncertainty.abs_value
   unc_accessor.Uncertainty.err_corr
   unc_accessor.Uncertainty.is_random
   unc_accessor.Uncertainty.is_structured
   unc_accessor.Uncertainty.is_systematic
   unc_accessor.Uncertainty.pdf_shape
   unc_accessor.Uncertainty.units
   unc_accessor.Uncertainty.value
   unc_accessor.Uncertainty.var_units
   unc_accessor.Uncertainty.var_value


Flag functions
==============

.. autosummary::
   :toctree: generated/

   flag_accessor.FlagAccessor
   flag_accessor.FlagAccessor.keys
   flag_accessor.FlagAccessor.__getitem__
   flag_accessor.FlagAccessor.__setitem__
   flag_accessor.FlagAccessor.data_vars
   flag_accessor.FlagAccessor.flag_vars
   flag_accessor.FlagVariable
   flag_accessor.FlagVariable.keys
   flag_accessor.FlagVariable.__getitem__
   flag_accessor.FlagVariable.__setitem__
   flag_accessor.FlagVariable.__delitem__
   flag_accessor.Flag
   flag_accessor.Flag.__getitem__
   flag_accessor.Flag.__setitem__
   flag_accessor.Flag.value
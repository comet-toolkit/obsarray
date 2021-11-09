"""
Handling for multiple ds templates
"""

from typing import Optional, Dict, List
import xarray
from obsarray import create_ds


__author__ = "Sam Hunt <sam.hunt@npl.co.uk>"


class DSTemplater:
    """
    Class to generate ``xarray.Dataset``'s from a set of templates.

    :param templates: template dictionary for each product format
    :param metadata_defs: metadata for each product format

    Each dictionary has as keys ``"template_name"`` strings that define the names of the different available dataset
    templates, e.g. "Level-0".

    For the ``templates`` dictionary the corresponding entries should be a `variable definition dictionary <https://dsbuilder.readthedocs.io/en/latest/content/quickstart.html#defining-a-template-dataset>`_.

    For ``metadata_defs`` the corresponding entries should be a dictionary of per product metadata.
    """

    def __init__(
        self, templates: Optional[Dict] = None, metadata_defs: Optional[Dict] = None
    ):
        self.templates = templates if templates is not None else {}
        self.metadata_defs = metadata_defs if metadata_defs is not None else {}

    def create(self, template_name: str, size: Dict[str, int]) -> xarray.Dataset:
        """
        Returns template dataset

        :param template_name: name of template to create (value returned by ``self.return_template_names()``)
        :param size: entry per dataset dimension with value of size

        :returns: Empty dataset
        """

        # Find template
        if template_name in self.return_template_names():
            template = self.templates[template_name]
        else:
            raise NameError(
                "No template: "
                + template_name
                + " - must be one of "
                + str(self.return_template_names())
            )

        # Find metadata def
        metadata = (
            self.metadata_defs[template_name]
            if template_name in self.metadata_defs.keys()
            else None
        )

        return create_ds(template, size, metadata=metadata)

    def return_template_names(self) -> list:
        """
        Returns available ds template names

        :returns: template names
        """

        return list(self.templates.keys())

    def return_template_var_names(self, template_name: str) -> List[str]:
        """
        Returns variables for specified template

        :param template_name: template name (value returned by ``self.return_template_names()``)

        :returns: variable names for specified template
        """

        return list(self.templates[template_name].keys())

    def return_ds_format_variable_dict(
        self, template_name: str, variable_name: str
    ) -> Dict:
        """
        Returns variable definition info for specified template variable

        :param template_name: template name (value returned by ``self.return_template_names()``)
        :param variable_name: variable name

        :returns: variable definition info
        """

        return self.templates[template_name][variable_name]

    def return_template_dim_names(self, template_name: str) -> List[str]:
        """
        Returns dims required for specified template

        :param template_name: template name (value returned by ``self.return_template_names()``)

        :returns: template dimensions
        """

        template = self.templates[template_name]

        template_dims = set()

        for var_name in template.keys():
            template_dims.update(template[var_name]["dim"])

        return list(template_dims)

    def create_size_dict(self, ds_format: str) -> Dict[str, None]:
        """
        Returns empty size dictionary for specified template

        :param template_name: template name (value returned by ``self.return_template_names()``)

        :return: empty sizes dictionary
        """

        dim_sizes_dict = dict()
        for dim in self.return_template_dim_names(ds_format):
            dim_sizes_dict[dim] = None

        return dim_sizes_dict


if __name__ == "__main__":
    pass

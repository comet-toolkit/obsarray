"""
dsdoc - module for documenting defined ds formats
"""

from obsarray import DSTemplater
import os
import numpy as np
from obsarray.templater.dataset_util import DatasetUtil


__author__ = "Sam Hunt <sam.hunt@npl.co.uk>"


TEXT = {
    "title": "Product Data Format",
    "section_title-introduction": "Introduction",
    "paragraph-introduction": "This document aims to specify definitions, conventions and formats of the various data "
    "products generated in the Hypernets land and water network processors.",
    "section_title-formats": "Product Definitions",
    "paragraph-formats": "Text",
    "section_title-dimensions": "Dimension",
    "paragraph-dimensions": "Text",
    "section_title-variables": "Variables Definition",
    "paragraph-variables": "Text",
    "subsection_title-fmt_variables": "<fmt> Variables",
    "paragraph-fmt_variables": "<fmt> Text",
    "section_title-metadata": "Metadata",
    "paragraph-metadata": "Text",
    "subsection_title-fmt_metadata": "<fmt> Metadata",
    "paragraph-fmt_metadata": "<fmt> Text",
}


class RSTDocWriter:
    def __init__(self):
        self.doc = []

    def write_doc(self, path):
        with open(path, "w") as f:
            for elem in self.doc:
                f.write(elem + "\n")

    def add_table(self, rows, title, header_widths=None, stub_columns=None):
        self.doc.append(
            self.create_table(
                rows, title, header_widths=header_widths, stub_columns=stub_columns
            )
        )

    def add_heading(self, title):
        self.doc.append(self.create_heading(title))

    def add_subheading(self, title):
        self.doc.append(self.create_subheading(title))

    def add_subsubheading(self, title):
        self.doc.append(self.create_subsubheading(title))

    def add_paragraph(self, paragraph):
        self.doc.append(paragraph + "\n")

    def create_table(self, rows, title, header_widths=None, stub_columns=None):

        rst_table = ".. list-table:: " + str(title) + "\n" "   :header-rows: 1\n"

        if header_widths is not None:
            widths_string = self.fmt_list_to_str(header_widths)
            rst_table += "   :widths: " + widths_string + "\n"

        if stub_columns is not None:
            rst_table += "   :stub-columns: " + str(stub_columns) + "\n"

        rst_table += "\n"

        for row in rows:

            if type(row) != list:
                row = [row]

            for i, column_value in enumerate(row):
                if i == 0:
                    rst_table += "   * - " + str(column_value) + "\n"
                else:
                    rst_table += "     - " + str(column_value) + "\n"

        return rst_table

    def _create_rst_heading(self, title, character="="):

        rst_header = title + "\n"
        rst_header += character * len(title) + "\n"

        return rst_header

    def create_heading(self, title):
        return self._create_rst_heading(title, character="=")

    def create_subheading(self, title):
        return self._create_rst_heading(title, character="-")

    def create_subsubheading(self, title):
        return self._create_rst_heading(title, character="~")

    def fmt_list_to_str(self, lst):

        lst_str = str(lst)
        lst_str = lst_str.replace("[", "")
        lst_str = lst_str.replace("]", "")
        lst_str = lst_str.replace("\\", "")
        lst_str = lst_str.replace("'", "")
        return lst_str


class DSDoc(RSTDocWriter):
    def __init__(self, dsb, text=TEXT):

        self.dsb = dsb
        self.text = text
        super().__init__()

    def build_dsdoc(self):

        self.add_heading(self.text["title"])

        # Add introduction

        self.add_subheading(self.text["section_title-introduction"])
        self.add_paragraph(self.text["paragraph-introduction"])

        # Add format

        self.add_subheading(self.text["section_title-formats"])
        self.add_paragraph(self.text["paragraph-formats"])

        fmt_bullets = ""
        for fmt in self.dsb.return_ds_formats():
            fmt_bullets += "* " + fmt + "\n"

        self.add_paragraph(fmt_bullets)

        # Add dimensions section

        self.add_subheading(self.text["section_title-dimensions"])
        self.add_paragraph(self.text["paragraph-dimensions"])
        self.add_dims_table()

        # Add variables section

        self.add_subheading(self.text["section_title-variables"])
        self.add_paragraph(self.text["paragraph-variables"])

        # Add variables subsections per ds_format
        for fmt in self.dsb.return_ds_formats():
            self.add_subsubheading(
                TEXT["subsection_title-fmt_variables"].replace("<fmt>", fmt)
            )
            self.add_paragraph(TEXT["paragraph-fmt_variables"].replace("<fmt>", fmt))

            self.add_ds_format_var_tables(fmt)

        # Add metadata section
        self.add_subheading(TEXT["section_title-metadata"])
        self.add_paragraph(TEXT["paragraph-metadata"])

        for fmt in self.dsb.return_ds_formats():
            self.add_subsubheading(
                TEXT["subsection_title-fmt_metadata"].replace("<fmt>", fmt)
            )
            self.add_paragraph(TEXT["paragraph-fmt_metadata"].replace("<fmt>", fmt))

    def add_dims_table(self):

        rows = [["Format", "Dimensions"]]

        for ds_format in self.dsb.return_ds_formats():
            dims = str(sorted(self.dsb.return_ds_format_dim_names(ds_format)))
            dims_str = self.fmt_list_to_str(dims)

            rows.append([ds_format, dims_str])

        return self.add_table(rows, "Dimensions")

    def add_ds_format_var_summary_table(self, ds_format):

        rows = [["Variable Name", "Standard Name", "Data Type", "Dimension"]]

        for variable_name in self.dsb.return_ds_format_variable_names(ds_format):
            variable_dict = self.dsb.return_ds_format_variable_dict(
                ds_format, variable_name
            )

            var_dim_str = self.fmt_list_to_str(variable_dict["dim"])
            var_dtype = self.fmt_dtype_to_str(variable_dict["dtype"])

            var_std_name = None
            if "attributes" in variable_dict.keys():
                var_std_name = (
                    variable_dict["attributes"]["standard_name"]
                    if "standard_name" in variable_dict["attributes"]
                    else ""
                )

            rows.append([variable_name, var_std_name, var_dtype, var_dim_str])

        return self.add_table(rows, ds_format + " variables")

    def add_var_table(self, variable_name, ds_format):

        variable_dict = self.dsb.return_ds_format_variable_dict(
            ds_format, variable_name
        )

        rows = [[variable_name, "Attribute", "Value"]]

        fill_value_str = (
            str(variable_dict["fill_value"])
            if "fill_value" in variable_dict
            else str(DatasetUtil.get_default_fill_value(variable_dict["dtype"]))
        )
        dtype_str = self.fmt_dtype_to_str(variable_dict["dtype"])
        var_dim_str = self.fmt_list_to_str(variable_dict["dim"])

        rows.append(["", "dim", var_dim_str])
        rows.append(["", "dtype", dtype_str])

        if "attributes" in variable_dict.keys():
            for attribute in variable_dict["attributes"].keys():
                rows.append(
                    ["", attribute, str(variable_dict["attributes"][attribute])]
                )

        rows.append(["", "_FillValue", fill_value_str])

        if "encoding" in variable_dict.keys():
            for encode in variable_dict["encoding"]:

                encode_str = str(variable_dict["encoding"][encode])
                if encode == "dtype":
                    encode_str = self.fmt_dtype_to_str(
                        variable_dict["encoding"][encode]
                    )

                rows.append(["", "encoding" + "_" + encode, encode_str])

        return self.add_table(rows, variable_name + " definition", stub_columns=1)

    def add_ds_format_var_tables(self, ds_format):

        self.add_ds_format_var_summary_table(ds_format)

        for variable_name in self.dsb.return_ds_format_variable_names(ds_format):
            self.add_var_table(variable_name, ds_format)

    def fmt_dtype_to_str(self, dtype):

        if dtype == np.int8:
            return "int8"
        if dtype == np.uint8:
            return "uint8"
        elif dtype == np.int16:
            return "int16"
        elif dtype == np.uint16:
            return "uint16"
        elif dtype == np.int32:
            return "int32"
        elif dtype == np.uint32:
            return "uint32"
        elif dtype == np.int64:
            return "int64"
        elif dtype == np.float32:
            return "float32"
        elif dtype == np.float64:
            return "float64"
        elif dtype == int:
            return "int"
        elif dtype == float:
            return "float"


def dsdoc_main(path, variables_defs, metadata_defs, custom_text=None):

    # Setup
    dsb = DSBuilder(variables_defs=variables_defs, metadata_defs=metadata_defs)

    if custom_text is not None:
        TEXT.update(custom_text)

    dsdoc = DSDoc(dsb, text=TEXT)

    # Build docs
    dsdoc.build_dsdoc()

    # Write docs
    outdir = os.path.dirname(path)
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    dsdoc.write_doc(os.path.join(outdir, "format.rst"))


if __name__ == "__main__":
    pass

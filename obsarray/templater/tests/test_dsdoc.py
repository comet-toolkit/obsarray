# """
# Tests for DatasetUtil class
# """
#
# import unittest
# from unittest.mock import patch, call
# from obsarray import DSTemplater
# from obsarray.templater.dsdoc import RSTDocWriter, DSDoc, dsdoc_main
# import numpy as np
#
# """___Authorship___"""
# __author__ = "Sam Hunt"
# __created__ = "17/9/2020"
#
#
# TEST_TEXT = {
#     "title": "Product Data Format",
#     "section_title-introduction": "Introduction",
#     "paragraph-introduction": "Text",
#     "section_title-formats": "Formats",
#     "paragraph-formats": "Text",
#     "section_title-dimensions": "Dimension",
#     "paragraph-dimensions": "Text",
#     "section_title-variables": "Variables Definition",
#     "paragraph-variables": "Text",
#     "subsection_title-fmt_variables": "<fmt> Variables",
#     "paragraph-fmt_variables": "<fmt> Text",
#     "section_title-metadata": "Metadata",
#     "paragraph-metadata": "Text",
#     "subsection_title-fmt_metadata": "<fmt> Metadata",
#     "paragraph-fmt_metadata": "<fmt> Text",
# }
#
# array_variable1_rows = [
#     ["array_variable1", "Attribute", "Value"],
#     ["", "dim", "dim1, dim2"],
#     ["", "dtype", "float32"],
#     ["", "standard_name", "array_variable_std_name1"],
#     ["", "long_name", "array_variable_long_name1"],
#     ["", "units", "units"],
#     ["", "preferred_symbol", "av"],
#     ["", "_FillValue", "9.96921e+36"],
#     ["", "encoding_dtype", "uint16"],
#     ["", "encoding_scale_factor", "1.0"],
#     ["", "encoding_offset", "0.0"],
# ]
#
# array_variable2_rows = [
#     ["array_variable2", "Attribute", "Value"],
#     ["", "dim", "dim3, dim4"],
#     ["", "dtype", "float32"],
#     ["", "standard_name", "array_variable_std_name2"],
#     ["", "long_name", "array_variable_long_name2"],
#     ["", "units", "units"],
#     ["", "preferred_symbol", "av"],
#     ["", "_FillValue", "9.96921e+36"],
#     ["", "encoding_dtype", "uint16"],
#     ["", "encoding_scale_factor", "1.0"],
#     ["", "encoding_offset", "0.0"],
# ]
#
#
# def setup_variables_defs():
#     test_variables = {
#         "array_variable1": {
#             "dim": ["dim1", "dim2"],
#             "dtype": np.float32,
#             "attributes": {
#                 "standard_name": "array_variable_std_name1",
#                 "long_name": "array_variable_long_name1",
#                 "units": "units",
#                 "preferred_symbol": "av",
#             },
#             "encoding": {"dtype": np.uint16, "scale_factor": 1.0, "offset": 0.0},
#         },
#         "array_variable2": {
#             "dim": ["dim3", "dim4"],
#             "dtype": np.float32,
#             "attributes": {
#                 "standard_name": "array_variable_std_name2",
#                 "long_name": "array_variable_long_name2",
#                 "units": "units",
#                 "preferred_symbol": "av",
#             },
#             "encoding": {"dtype": np.uint16, "scale_factor": 1.0, "offset": 0.0},
#         },
#     }
#
#     return {"def1": test_variables}
#
#
# def setup_metadata_defs():
#     return {}
#
#
# def setup_dsb(
#     templates=setup_variables_defs(), metadata_defs=setup_metadata_defs()
# ):
#     return DSTemplater(templates=templates, metadata_defs=metadata_defs)
#
#
# class TestRSTDocWriter(unittest.TestCase):
#     def test_create_table(self):
#         dw = RSTDocWriter()
#
#         rows = [["test1", "test2", "test3"], ["test4", "test5", "test6"]]
#         title = "table title"
#         spacing = [30, 40, 30]
#
#         tbl = dw.create_table(rows, title, spacing)
#         expected_tbl = (
#             ".. list-table:: table title\n"
#             "   :header-rows: 1\n"
#             "   :widths: 30 40 30\n\n"
#             "   * - test1\n"
#             "     - test2\n"
#             "     - test3\n"
#             "   * - test4\n"
#             "     - test5\n"
#             "     - test6\n"
#         )
#
#         self.assertEqual(tbl, expected_tbl)
#
#     def test_create_table_single_column(self):
#         dw = RSTDocWriter()
#
#         rows = ["test1", "test2", "test3"]
#         title = "table title"
#
#         tbl = dw.create_table(rows, title)
#         expected_tbl = (
#             ".. list-table:: table title\n"
#             "   :header-rows: 1\n\n"
#             "   * - test1\n"
#             "   * - test2\n"
#             "   * - test3\n"
#         )
#
#         self.assertEqual(tbl, expected_tbl)
#
#     def test__create_rst_heading(self):
#         dw = RSTDocWriter()
#
#         heading = dw._create_rst_heading("text", "=")
#         expected_heading = "text\n" "====\n"
#
#         self.assertEqual(heading, expected_heading)
#
#     @patch("templater.dsdoc.RSTDocWriter.create_table")
#     def test_add_table(self, mock):
#         dw = RSTDocWriter()
#         dw.add_table("rows", "title", "widths")
#         mock.assert_called_once_with("rows", "title", header_widths="widths")
#
#         self.assertEqual(dw.doc[0], mock.return_value)
#
#     @patch("templater.dsdoc.RSTDocWriter._create_rst_heading")
#     def test_create_heading(self, mock):
#         dw = RSTDocWriter()
#         heading = dw.create_heading("text")
#         mock.assert_called_once_with("text", character="=")
#
#     @patch("templater.dsdoc.RSTDocWriter._create_rst_heading")
#     def test_create_subheading(self, mock):
#         dw = RSTDocWriter()
#         heading = dw.create_subheading("text")
#         mock.assert_called_once_with("text", character="-")
#
#     @patch("templater.dsdoc.RSTDocWriter._create_rst_heading")
#     def test_create_subsubheading(self, mock):
#         dw = RSTDocWriter()
#         heading = dw.create_subsubheading("text")
#         mock.assert_called_once_with("text", character="~")
#
#     @patch("templater.dsdoc.RSTDocWriter.create_heading")
#     def test_add_heading(self, mock):
#         dw = RSTDocWriter()
#         dw.add_heading("text")
#         mock.assert_called_once_with("text")
#
#         self.assertEqual(dw.doc[0], mock.return_value)
#
#     @patch("templater.dsdoc.RSTDocWriter.create_subheading")
#     def test_add_subheading(self, mock):
#         dw = RSTDocWriter()
#         heading = dw.add_subheading("text")
#         mock.assert_called_once_with("text")
#
#         self.assertEqual(dw.doc[0], mock.return_value)
#
#     @patch("templater.dsdoc.RSTDocWriter.create_subsubheading")
#     def test_add_subsubheading(self, mock):
#         dw = RSTDocWriter()
#         dw.add_subsubheading("text")
#         mock.assert_called_once_with("text")
#
#         self.assertEqual(dw.doc[0], mock.return_value)
#
#     def test_add_paragraph(self):
#         dw = RSTDocWriter()
#         dw.add_paragraph("text")
#
#         self.assertEqual(dw.doc[0], "text" + "\n")
#
#
# class TestDSDoc(unittest.TestCase):
#     @patch("templater.dsdoc.RSTDocWriter.add_table")
#     def test_build_dims_table(self, mock):
#
#         dsb = setup_dsb()
#         dsdoc = DSDoc(dsb)
#
#         dsdoc.add_dims_table()
#
#         expected_rows = [["Format", "Dimensions"], ["def1", "dim1, dim2, dim3, dim4"]]
#         expected_title = "Dimensions"
#
#         mock.assert_called_once_with(expected_rows, expected_title)
#
#     @patch("templater.dsdoc.DSDoc.add_var_table")
#     def test_add_ds_format_var_tables(self, mock):
#         dsb = setup_dsb()
#         dsdoc = DSDoc(dsb)
#
#         dsdoc.add_ds_format_var_tables("def1")
#
#         calls = [call("array_variable1", "def1"), call("array_variable2", "def1")]
#
#         self.assertCountEqual(mock.call_args_list, calls)
#
#     @patch("templater.dsdoc.RSTDocWriter.add_table")
#     def test_build_var_table(self, mock):
#         dsb = setup_dsb()
#         dsdoc = DSDoc(dsb)
#
#         dsdoc.add_var_table("array_variable1", "def1")
#
#         expected_rows = array_variable1_rows
#         expected_title = "array_variable1 definition"
#
#         mock.assert_called_once_with(expected_rows, expected_title, stub_columns=1)
#
#     @patch("templater.dsdoc.RSTDocWriter.add_table")
#     def test_add_ds_format_var_summary_table(self, mock):
#         dsb = setup_dsb()
#         dsdoc = DSDoc(dsb)
#
#         dsdoc.add_ds_format_var_summary_table("def1")
#
#         expected_rows = [
#             ["Variable Name", "Standard Name", "Data Type", "Dimension"],
#             ["array_variable1", "array_variable_std_name1", "float32", "dim1, dim2"],
#             ["array_variable2", "array_variable_std_name2", "float32", "dim3, dim4"],
#         ]
#         expected_title = "def1 variables"
#
#         mock.assert_called_once_with(expected_rows, expected_title)
#
#     def test_build_dsdoc(self):
#
#         dsb = setup_dsb()
#         dsdoc = DSDoc(dsb, text=TEST_TEXT)
#
#         dsdoc.build_dsdoc()
#
#         dsdoc_mk = DSDoc(dsb, text=TEST_TEXT)
#         dsdoc_mk.add_ds_format_var_summary_table("def1")
#         dsdoc_mk.add_dims_table()
#         def1_summary_tbl = dsdoc_mk.doc[0]
#         dims_tbl = dsdoc_mk.doc[1]
#
#         expected_doc = [
#             "Product Data Format\n===================\n",
#             "Formats\n-------\n",
#             "Text\n",
#             "* def1\n",
#             "Introduction\n------------\n",
#             "Text\n",
#             "Dimension\n---------\n",
#             "Text\n",
#             dims_tbl,
#             "Variables Definition\n--------------------\n",
#             "Text\n",
#             "def1 Variables\n~~~~~~~~~~~~~~\n",
#             "def1 Text\n",
#             def1_summary_tbl,
#             dsdoc.create_table(
#                 array_variable1_rows, "array_variable1 definition", stub_columns=1
#             ),
#             dsdoc.create_table(
#                 array_variable2_rows, "array_variable2 definition", stub_columns=1
#             ),
#             "Metadata\n--------\n",
#             "Text\n",
#             "def1 Metadata\n~~~~~~~~~~~~~\n",
#             "def1 Text\n",
#         ]
#
#         self.assertCountEqual(expected_doc, dsdoc.doc)
#
#     def test_write_dsdoc(self):
#
#         dsb = setup_dsb()
#         dsdoc = DSDoc(dsb, text=TEST_TEXT)
#
#         dsdoc.build_dsdoc()
#         dsdoc.write_doc("dsdoc/src/ds.rst")
#
#
# if __name__ == "__main__":
#     unittest.main()

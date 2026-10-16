import numpy as np
from matplotlib.figure import Figure

from analysis.shared_reporting_utils.quick_look_plots import (
    quick_look_sections,
    unsupported_shape_message,
)
from susi.io.load_output_data import NetcdfVariableArray, NetcdfVariablePath


def _make_variable_array(shape: tuple[int, ...]) -> NetcdfVariableArray:
    return NetcdfVariableArray(raw=np.zeros(shape))


class TestUnsupportedShapeMessage:
    def test_message_names_the_actual_shape(self):
        message = unsupported_shape_message((5, 7))
        assert "(5, 7)" in message
        assert "3D" in message


class TestQuickLookSections:
    def test_3d_variable_yields_no_message_and_two_titled_figures(self):
        variables_values = {
            NetcdfVariablePath("/stand/volume"): _make_variable_array((1, 3, 4)),
        }

        sections = list(quick_look_sections(variables_values))

        assert len(sections) == 1
        var_path, message, figures = sections[0]
        assert var_path == NetcdfVariablePath("/stand/volume")
        assert message is None
        assert [title for title, _ in figures] == [
            "Time-Space Waterfall (Grouped Bars)",
            "Spatial Statistics (Mean ± Std)",
        ]
        assert all(isinstance(fig, Figure) for _, fig in figures)

    def test_non_3d_variable_yields_message_and_no_figures(self):
        variables_values = {
            NetcdfVariablePath("/stand/scalar_thing"): _make_variable_array((5, 7)),
        }

        sections = list(quick_look_sections(variables_values))

        assert len(sections) == 1
        var_path, message, figures = sections[0]
        assert var_path == NetcdfVariablePath("/stand/scalar_thing")
        assert message == unsupported_shape_message((5, 7))
        assert figures == []

    def test_preserves_input_order_across_multiple_variables(self):
        variables_values = {
            NetcdfVariablePath("/a"): _make_variable_array((5,)),
            NetcdfVariablePath("/b"): _make_variable_array((1, 3, 4)),
            NetcdfVariablePath("/c"): _make_variable_array((5, 7)),
        }

        sections = list(quick_look_sections(variables_values))

        assert [var_path for var_path, _, _ in sections] == [
            NetcdfVariablePath("/a"),
            NetcdfVariablePath("/b"),
            NetcdfVariablePath("/c"),
        ]

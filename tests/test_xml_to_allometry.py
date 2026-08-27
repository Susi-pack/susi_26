from hypothesis import given
from hypothesis import strategies as st
import pytest

from tools import xml_to_allometry


# %% out_of_range_message


@pytest.mark.parametrize(
    "value,min_value,max_value",
    [
        (0, 0, 1000),  # lower bound, inclusive
        (1000, 0, 1000),  # upper bound, inclusive
        (500, 0, 1000),  # comfortably inside
    ],
)
def test_out_of_range_message_within_range_returns_none(value, min_value, max_value):
    assert xml_to_allometry.out_of_range_message("altitude", value, min_value, max_value) is None


def test_out_of_range_message_below_range_returns_message():
    message = xml_to_allometry.out_of_range_message("altitude", -1, 0, 1000)
    assert message is not None
    assert "altitude" in message
    assert "-1" in message


def test_out_of_range_message_above_range_returns_message():
    message = xml_to_allometry.out_of_range_message("ddy", 2001, 500, 2000)
    assert message is not None
    assert "ddy" in message
    assert "2001" in message


# %% parse_CLI_arguments


@pytest.fixture
def dummy_xml_file(tmp_path):
    xml_path = tmp_path / "stand.xml"
    xml_path.write_text("<ForestPropertyData></ForestPropertyData>")
    return xml_path


@pytest.fixture
def output_dir(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    return out


def _run_parse_CLI_arguments(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["xml_to_allometry.py", *argv])
    return xml_to_allometry.parse_CLI_arguments()


def test_parse_CLI_arguments_requires_altitude(monkeypatch, dummy_xml_file, output_dir, capsys):
    with pytest.raises(SystemExit):
        _run_parse_CLI_arguments(
            monkeypatch,
            [str(dummy_xml_file), str(output_dir), "--ddy=1200"],
        )
    assert "--altitude" in capsys.readouterr().err


def test_parse_CLI_arguments_requires_ddy(monkeypatch, dummy_xml_file, output_dir, capsys):
    with pytest.raises(SystemExit):
        _run_parse_CLI_arguments(
            monkeypatch,
            [str(dummy_xml_file), str(output_dir), "--altitude=150"],
        )
    assert "--ddy" in capsys.readouterr().err


def test_parse_CLI_arguments_blocks_out_of_range_altitude_by_default(
    monkeypatch, dummy_xml_file, output_dir, capsys
):
    with pytest.raises(SystemExit):
        _run_parse_CLI_arguments(
            monkeypatch,
            [str(dummy_xml_file), str(output_dir), "--altitude=1500", "--ddy=1200"],
        )
    stderr = capsys.readouterr().err
    assert "altitude" in stderr
    assert "--allow-out-of-range-values" in stderr


def test_parse_CLI_arguments_blocks_out_of_range_ddy_by_default(
    monkeypatch, dummy_xml_file, output_dir, capsys
):
    with pytest.raises(SystemExit):
        _run_parse_CLI_arguments(
            monkeypatch,
            [str(dummy_xml_file), str(output_dir), "--altitude=150", "--ddy=2500"],
        )
    stderr = capsys.readouterr().err
    assert "ddy" in stderr
    assert "--allow-out-of-range-values" in stderr


def test_parse_CLI_arguments_allows_out_of_range_with_override(
    monkeypatch, dummy_xml_file, output_dir, capsys
):
    cli_args = _run_parse_CLI_arguments(
        monkeypatch,
        [
            str(dummy_xml_file),
            str(output_dir),
            "--altitude=1500",
            "--ddy=1200",
            "--allow-out-of-range-values",
        ],
    )
    assert cli_args.altitude == 1500
    assert cli_args.ddy == 1200
    assert "Warning" in capsys.readouterr().out


def test_parse_CLI_arguments_reports_both_out_of_range_values_together(
    monkeypatch, dummy_xml_file, output_dir, capsys
):
    # Both altitude and ddy are out of range: the user should learn about both
    # in one run, rather than fixing one and being told about the other on retry.
    with pytest.raises(SystemExit):
        _run_parse_CLI_arguments(
            monkeypatch,
            [str(dummy_xml_file), str(output_dir), "--altitude=1500", "--ddy=2500"],
        )
    stderr = capsys.readouterr().err
    # Values, not just flag names, must appear: the usage line always mentions
    # both flag names, so checking for "altitude"/"ddy" alone would pass even
    # if only one violation were actually reported.
    assert "1500" in stderr
    assert "2500" in stderr


def test_parse_CLI_arguments_rejects_nan_altitude_even_with_override(
    monkeypatch, dummy_xml_file, output_dir, capsys
):
    with pytest.raises(SystemExit):
        _run_parse_CLI_arguments(
            monkeypatch,
            [
                str(dummy_xml_file),
                str(output_dir),
                "--altitude=nan",
                "--ddy=1200",
                "--allow-out-of-range-values",
            ],
        )
    assert "altitude" in capsys.readouterr().err


def test_parse_CLI_arguments_rejects_nan_ddy_even_with_override(
    monkeypatch, dummy_xml_file, output_dir, capsys
):
    with pytest.raises(SystemExit):
        _run_parse_CLI_arguments(
            monkeypatch,
            [
                str(dummy_xml_file),
                str(output_dir),
                "--altitude=150",
                "--ddy=nan",
                "--allow-out-of-range-values",
            ],
        )
    assert "ddy" in capsys.readouterr().err


def test_parse_CLI_arguments_accepts_in_range_values_without_warning(
    monkeypatch, dummy_xml_file, output_dir, capsys
):
    cli_args = _run_parse_CLI_arguments(
        monkeypatch,
        [str(dummy_xml_file), str(output_dir), "--altitude=150", "--ddy=1200"],
    )
    assert cli_args.altitude == 150
    assert cli_args.ddy == 1200
    assert capsys.readouterr().out == ""


@given(
    coords=st.lists(
        st.tuples(
            st.floats(allow_nan=False, allow_infinity=False),
            st.floats(allow_nan=False, allow_infinity=False),
        ),
        min_size=1,
    )
)
def test_parse_coordinates_roundtrip(coords):
    # Build the input string in the same format the function expects
    input_str = " ".join(f"{x},{y}" for x, y in coords)

    result = xml_to_allometry.parse_polygon_to_coords(input_str)

    # Check structure
    assert len(result) == len(coords)
    for (rx, ry), (ox, oy) in zip(result, coords):
        assert rx == pytest.approx(ox)
        assert ry == pytest.approx(oy)

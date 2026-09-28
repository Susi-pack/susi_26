import json
import pytest
import xmltodict
from hypothesis import given
from hypothesis import strategies as st
from pyproj import Transformer
from shapely.geometry import Polygon

from susi.io.load_output_data import StandID
from susi.io.utils import SRC_DIR
from susi.io.susi_parameter_model import CanopyLayerName, read_allometry_info_from_csv
from tools.shared_allometry_tool_utils import input_validation
from susi.io.stand_data import (
    SOURCE_CRS,
    centroid_to_ykj,
    point_to_ykj,
    dump_stand_data_document as shared_dump_stand_data_document,
    load_stand_data_document_from_json,
)
from tools.xml_to_allometry import xml_to_allometry


# %% out_of_range_message reuse (from the shared package)


def test_out_of_range_message_reused_from_shared_package():
    assert input_validation.out_of_range_message("altitude", 500, 0, 1000) is None
    message = input_validation.out_of_range_message("altitude", -1, 0, 1000)
    assert message is not None
    assert "altitude" in message


# %% XmlConfig
#
# XmlConfig is a StrictFrozenModel (susi.io.extra_pydantic_types): presence/
# unknown-field checking, required-vs-defaulted fields, and extra="forbid"
# all come from Pydantic itself, the same as metsakeskus_to_allometry.py's
# ExtractionConfig and new_growth_allometry.py's NewGrowthConfig.


def test_xml_config_requires_altitude():
    with pytest.raises(ValueError, match="altitude"):
        xml_to_allometry.XmlConfig.model_validate({"ddy": 1200.0})


def test_xml_config_requires_ddy():
    with pytest.raises(ValueError, match="ddy"):
        xml_to_allometry.XmlConfig.model_validate({"altitude": 150.0})


def test_xml_config_reports_all_missing_fields_together():
    with pytest.raises(ValueError) as exc_info:
        xml_to_allometry.XmlConfig.model_validate({})
    message = str(exc_info.value)
    assert "altitude" in message
    assert "ddy" in message


def test_xml_config_rejects_unknown_field():
    with pytest.raises(ValueError, match="typo_field"):
        xml_to_allometry.XmlConfig.model_validate(
            {"altitude": 150.0, "ddy": 1200.0, "typo_field": 1}
        )


def test_xml_config_applies_defaults():
    config = xml_to_allometry.XmlConfig.model_validate(
        {"altitude": 150.0, "ddy": 1200.0}
    )
    assert config.n_trees == 20
    assert config.start_year == 5
    assert config.end_year == 80
    assert config.step_years == 5


def test_xml_config_overrides_defaults():
    config = xml_to_allometry.XmlConfig.model_validate(
        {"altitude": 150.0, "ddy": 1200.0, "n_trees": 5, "step_years": 10}
    )
    assert config.n_trees == 5
    assert config.step_years == 10


def test_xml_config_is_frozen():
    config = xml_to_allometry.XmlConfig(altitude=150.0, ddy=1200.0)
    with pytest.raises(Exception):  # noqa: B017 -- pydantic's frozen-model error
        setattr(config, "altitude", 200.0)  # noqa: B010


def test_load_xml_config_reads_toml(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text("altitude = 150.0\nddy = 1200.0\nn_trees = 15\n")
    config = xml_to_allometry.load_xml_config(config_path)
    assert config.altitude == 150.0
    assert config.n_trees == 15


# %% default_config.toml
#
# Guards against the shipped default/template config drifting from
# XmlConfig's own field defaults -- see that file's header comment.

DEFAULT_CONFIG_PATH = SRC_DIR / "tools" / "xml_to_allometry" / "default_config.toml"


def test_default_config_toml_optional_fields_match_model_defaults():
    config = xml_to_allometry.load_xml_config(DEFAULT_CONFIG_PATH)
    defaults = xml_to_allometry.XmlConfig(altitude=0.0, ddy=0.0)
    assert config.n_trees == defaults.n_trees
    assert config.start_year == defaults.start_year
    assert config.end_year == defaults.end_year
    assert config.step_years == defaults.step_years


def test_default_config_toml_required_fields_are_deliberately_out_of_range():
    """The shipped file's altitude/ddy placeholders must stay outside
    [ALTITUDE_MIN, ALTITUDE_MAX] / [DDY_MIN, DDY_MAX] -- that's what makes an
    untouched copy fail loudly instead of running silently."""
    config = xml_to_allometry.load_xml_config(DEFAULT_CONFIG_PATH)
    assert not (
        input_validation.ALTITUDE_MIN
        <= config.altitude
        <= input_validation.ALTITUDE_MAX
    )
    assert not (input_validation.DDY_MIN <= config.ddy <= input_validation.DDY_MAX)


# %% XML stand fixtures


def _polygon_coordinates() -> str:
    return "385000,6685000 385050,6685000 385050,6685050 385000,6685050"


def _hole_coordinates(offset: float = 0.0) -> str:
    """A 10x10 m hole strictly inside _polygon_coordinates' 50x50 m square
    (offset shifts it along x, so two holes don't overlap)."""
    x0, y0 = 385010 + offset, 6685010
    return f"{x0},{y0} {x0 + 10},{y0} {x0 + 10},{y0 + 10} {x0},{y0 + 10}"


def _stand_xml_block(
    stand_id: str,
    *,
    include_tree_strata: bool,
    include_second_species: bool = False,
    srs_name: str | None = SOURCE_CRS,
    exterior_coordinates: str | None = None,
    interior_coordinates: tuple[str, ...] = (),
) -> str:
    second_stratum_block = (
        """
            <tst:TreeStratum>
              <tst:TreeSpecies>2</tst:TreeSpecies>
              <tst:Age>35</tst:Age>
              <tst:BasalArea>8.0</tst:BasalArea>
              <tst:StemCount>300</tst:StemCount>
              <tst:MeanDiameter>15.0</tst:MeanDiameter>
              <tst:MeanHeight>14.0</tst:MeanHeight>
            </tst:TreeStratum>
        """
        if include_second_species
        else ""
    )
    tree_strata_block = (
        f"""
          <tst:TreeStrata>
            <tst:TreeStratum>
              <tst:TreeSpecies>1</tst:TreeSpecies>
              <tst:Age>40</tst:Age>
              <tst:BasalArea>20.0</tst:BasalArea>
              <tst:StemCount>500</tst:StemCount>
              <tst:MeanDiameter>20.0</tst:MeanDiameter>
              <tst:MeanHeight>18.0</tst:MeanHeight>
            </tst:TreeStratum>
            {second_stratum_block}
          </tst:TreeStrata>
        """
        if include_tree_strata
        else ""
    )
    srs_attribute = f' srsName="{srs_name}"' if srs_name is not None else ""
    interior_blocks = "".join(
        f"""
              <gml:interior>
                <gml:LinearRing>
                  <gml:coordinates>{coordinates}</gml:coordinates>
                </gml:LinearRing>
              </gml:interior>"""
        for coordinates in interior_coordinates
    )
    exterior_coordinates = exterior_coordinates or _polygon_coordinates()
    return f"""
    <st:Stand id="{stand_id}">
      <st:StandBasicData>
        <st:FertilityClass>3</st:FertilityClass>
        <st:MainGroup>1</st:MainGroup>
        <st:SubGroup>2</st:SubGroup>
        <st:Area>1.5</st:Area>
        <gdt:PolygonGeometry>
          <gml:polygonProperty>
            <gml:Polygon{srs_attribute}>
              <gml:exterior>
                <gml:LinearRing>
                  <gml:coordinates>{exterior_coordinates}</gml:coordinates>
                </gml:LinearRing>
              </gml:exterior>{interior_blocks}
            </gml:Polygon>
          </gml:polygonProperty>
        </gdt:PolygonGeometry>
      </st:StandBasicData>
      <ts:TreeStandData>
        <ts:TreeStandDataDate>
          <tss:TreeStandSummary>
            <tss:MeanDiameter>20.0</tss:MeanDiameter>
            <tss:MeanAge>40</tss:MeanAge>
            <tss:BasalArea>20.0</tss:BasalArea>
            <tss:MeanHeight>18.0</tss:MeanHeight>
            <tss:Volume>150.0</tss:Volume>
          </tss:TreeStandSummary>
          {tree_strata_block}
        </ts:TreeStandDataDate>
      </ts:TreeStandData>
    </st:Stand>
    """


def _forest_property_xml(stand_blocks: str) -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        "<ForestPropertyData>\n"
        "  <st:Stands>\n"
        f"{stand_blocks}"
        "  </st:Stands>\n"
        "</ForestPropertyData>\n"
    )


def _parsed_stand(
    stand_id: str,
    *,
    include_tree_strata: bool,
    include_second_species: bool = False,
    **polygon_options,
) -> dict:
    """One <st:Stand> parsed back into the dict shape
    get_stand_data_from_xml/build_stands expect (i.e. what
    read_stands_from_xml_file would hand them for a single stand).
    polygon_options go to _stand_xml_block (srs_name, exterior_coordinates,
    interior_coordinates)."""
    xml_text = _forest_property_xml(
        _stand_xml_block(
            stand_id,
            include_tree_strata=include_tree_strata,
            include_second_species=include_second_species,
            **polygon_options,
        )
    )
    parsed = xmltodict.parse(xml_text)
    return parsed["ForestPropertyData"]["st:Stands"]["st:Stand"]


# %% read_stands_from_xml_file


def test_read_stands_from_xml_file_wraps_a_single_stand_in_a_list(tmp_path):
    # xmltodict collapses a single repeated <st:Stand> element to a bare
    # dict instead of a one-item list. Exercised through the real parser on
    # actual XML text (not a hand-built list) so this catches the xmltodict
    # quirk itself, not just a mocked-out shape (#281).
    xml_path = tmp_path / "single_stand.xml"
    xml_path.write_text(
        _forest_property_xml(_stand_xml_block("1", include_tree_strata=True))
    )

    stands = xml_to_allometry.read_stands_from_xml_file(xml_path)

    assert isinstance(stands, list)
    assert len(stands) == 1


def test_read_stands_from_xml_file_keeps_multiple_stands_as_a_list(tmp_path):
    xml_path = tmp_path / "two_stands.xml"
    xml_path.write_text(
        _forest_property_xml(
            _stand_xml_block("1", include_tree_strata=True)
            + _stand_xml_block("2", include_tree_strata=True)
        )
    )

    stands = xml_to_allometry.read_stands_from_xml_file(xml_path)

    assert isinstance(stands, list)
    assert len(stands) == 2


def test_build_stands_handles_a_single_stand_file_end_to_end(tmp_path):
    # Regression test for #281: a real single-stand XML file used to crash
    # build_stands with "TypeError: string indices must be integers, not
    # 'str'" because xmltodict handed it a dict, not a list.
    xml_path = tmp_path / "single_stand.xml"
    xml_path.write_text(
        _forest_property_xml(_stand_xml_block("1", include_tree_strata=True))
    )

    stands = xml_to_allometry.read_stands_from_xml_file(xml_path)
    parsed_stands, skipped = xml_to_allometry.build_stands(stands)

    assert [ps.id for ps in parsed_stands] == [StandID("1")]
    assert skipped == []


# %% get_stand_data_from_xml / NoTreeStrataError / build_stands


def test_get_stand_data_from_xml_parses_a_full_stand():
    parsed_stand = xml_to_allometry.get_stand_data_from_xml(
        _parsed_stand("1", include_tree_strata=True)
    )
    assert parsed_stand.id == StandID("1")
    assert parsed_stand.fertility_class == 3
    assert parsed_stand.main_species == 1
    assert parsed_stand.tree_strata.pine.basal_area == pytest.approx(20.0)
    assert parsed_stand.soil_type is None


def test_get_stand_data_from_xml_parses_soil_type_when_present():
    xml_text = _forest_property_xml(
        _stand_xml_block("1", include_tree_strata=True)
    ).replace(
        "<st:Area>1.5</st:Area>", "<st:Area>1.5</st:Area>\n<st:SoilType>2</st:SoilType>"
    )
    parsed = xmltodict.parse(xml_text)["ForestPropertyData"]["st:Stands"]["st:Stand"]

    parsed_stand = xml_to_allometry.get_stand_data_from_xml(parsed)

    assert parsed_stand.soil_type == 2


def test_get_stand_data_from_xml_picks_main_species_by_largest_basal_area():
    parsed_stand = xml_to_allometry.get_stand_data_from_xml(
        _parsed_stand("1", include_tree_strata=True, include_second_species=True)
    )
    # species 1: basal_area=20.0, species 2: basal_area=8.0
    assert parsed_stand.main_species == 1


def test_get_stand_data_from_xml_stores_x_ykj_and_y_ykj():
    parsed_stand = xml_to_allometry.get_stand_data_from_xml(
        _parsed_stand("1", include_tree_strata=True)
    )
    assert isinstance(parsed_stand.x_ykj, int)
    assert isinstance(parsed_stand.y_ykj, int)


def test_get_stand_data_from_xml_builds_a_polygon_from_the_exterior_ring():
    parsed_stand = xml_to_allometry.get_stand_data_from_xml(
        _parsed_stand("1", include_tree_strata=True)
    )

    assert isinstance(parsed_stand.polygon, Polygon)
    assert parsed_stand.polygon.equals(
        Polygon(xml_to_allometry.parse_polygon_to_coords(_polygon_coordinates()))
    )
    assert list(parsed_stand.polygon.interiors) == []


def test_get_stand_data_from_xml_keeps_a_single_interior_ring_as_a_hole():
    # Exactly one <gml:interior>: xmltodict hands it over as a bare dict,
    # not a one-item list.
    parsed_stand = xml_to_allometry.get_stand_data_from_xml(
        _parsed_stand(
            "1", include_tree_strata=True, interior_coordinates=(_hole_coordinates(),)
        )
    )

    assert len(parsed_stand.polygon.interiors) == 1
    # 50x50 m square minus one 10x10 m hole.
    assert parsed_stand.polygon.area == pytest.approx(2500 - 100)


def test_get_stand_data_from_xml_keeps_several_interior_rings_as_holes():
    parsed_stand = xml_to_allometry.get_stand_data_from_xml(
        _parsed_stand(
            "1",
            include_tree_strata=True,
            interior_coordinates=(_hole_coordinates(0), _hole_coordinates(20)),
        )
    )

    assert len(parsed_stand.polygon.interiors) == 2
    assert parsed_stand.polygon.area == pytest.approx(2500 - 2 * 100)


def test_get_stand_data_from_xml_takes_ykj_from_the_polygon_centroid():
    # 20 km wide, so the first vertex (x=380 km) and the centroid (x=390 km)
    # land in different YKJ easting units: the old first-vertex choice would
    # give a different x_ykj.
    wide = "380000,6685000 400000,6685000 400000,6686000 380000,6686000"
    parsed_stand = xml_to_allometry.get_stand_data_from_xml(
        _parsed_stand("1", include_tree_strata=True, exterior_coordinates=wide)
    )

    assert (parsed_stand.x_ykj, parsed_stand.y_ykj) == centroid_to_ykj(
        parsed_stand.polygon
    )
    assert parsed_stand.x_ykj != point_to_ykj(380000, 6685000)[0]


def test_centroid_to_ykj_is_the_shared_function():
    assert xml_to_allometry.centroid_to_ykj is centroid_to_ykj


def _coordinates_in(crs: str) -> str:
    """_polygon_coordinates' square, reprojected from EPSG:3067 into crs and
    written back out as gml:coordinates pairs (x first, as the tool reads
    them)."""
    transformer = Transformer.from_crs(SOURCE_CRS, crs, always_xy=True)
    pairs = xml_to_allometry.parse_polygon_to_coords(_polygon_coordinates())
    return " ".join("{},{}".format(*transformer.transform(x, y)) for x, y in pairs)


@pytest.mark.parametrize(
    "srs_name",
    [pytest.param("EPSG:4326", id="WGS84"), pytest.param("EPSG:2393", id="YKJ")],
)
def test_get_stand_data_from_xml_reprojects_a_polygon_in_another_crs(srs_name):
    in_source_crs = xml_to_allometry.get_stand_data_from_xml(
        _parsed_stand("1", include_tree_strata=True)
    )
    in_other_crs = xml_to_allometry.get_stand_data_from_xml(
        _parsed_stand(
            "1",
            include_tree_strata=True,
            srs_name=srs_name,
            exterior_coordinates=_coordinates_in(srs_name),
        )
    )

    assert in_other_crs.polygon.equals_exact(in_source_crs.polygon, tolerance=1e-3)
    assert (in_other_crs.x_ykj, in_other_crs.y_ykj) == (
        in_source_crs.x_ykj,
        in_source_crs.y_ykj,
    )


@pytest.mark.parametrize(
    ("srs_name", "message"),
    [
        pytest.param(None, "no CRS", id="missing-srsName"),
        pytest.param("EPSG:999999", "EPSG:999999", id="unknown-srsName"),
    ],
)
def test_build_stands_fails_the_run_on_a_polygon_with_no_usable_crs(srs_name, message):
    # Nothing to reproject from is a whole-export problem: it must abort
    # build_stands, not turn into a per-stand StandSkipped the way a
    # missing TreeStrata does.
    stands = [
        _parsed_stand("1", include_tree_strata=True),
        _parsed_stand("2", include_tree_strata=True, srs_name=srs_name),
    ]

    with pytest.raises(ValueError, match=message) as error:
        xml_to_allometry.build_stands(stands)
    assert not isinstance(error.value, xml_to_allometry.NoTreeStrataError)
    assert "Stand 2" in str(error.value)


def test_get_stand_data_from_xml_raises_no_tree_strata_error_when_missing():
    with pytest.raises(xml_to_allometry.NoTreeStrataError, match="2"):
        xml_to_allometry.get_stand_data_from_xml(
            _parsed_stand("2", include_tree_strata=False)
        )


def test_build_stands_reports_skip_by_stand_id_and_reason():
    stands = [
        _parsed_stand("1", include_tree_strata=True),
        _parsed_stand("2", include_tree_strata=False),
    ]
    parsed_stands, skipped = xml_to_allometry.build_stands(stands)

    assert [ps.id for ps in parsed_stands] == [StandID("1")]
    assert len(skipped) == 1
    assert skipped[0].stand_id == StandID("2")
    assert "TreeStrata" in skipped[0].reason


def test_build_stands_keeps_every_stand_when_none_are_skipped():
    stands = [_parsed_stand("1", include_tree_strata=True)]
    parsed_stands, skipped = xml_to_allometry.build_stands(stands)
    assert len(parsed_stands) == 1
    assert skipped == []


# %% parse_polygon_to_coords


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


# %% plan_stand_output / process_stand / dump_stand_data_document


def _parsed_stand_data(
    stand_id="1", *, include_second_species=False
) -> xml_to_allometry.ParsedStand:
    return xml_to_allometry.get_stand_data_from_xml(
        _parsed_stand(
            stand_id,
            include_tree_strata=True,
            include_second_species=include_second_species,
        )
    )


def test_plan_stand_output_names_the_csv_by_stand_id(tmp_path):
    parsed_stand = _parsed_stand_data("42")
    assert xml_to_allometry.plan_stand_output(parsed_stand, tmp_path) == (
        tmp_path / "42.csv"
    )


def test_process_stand_writes_a_csv_round_tripping_through_the_real_reader(tmp_path):
    parsed_stand = _parsed_stand_data("1")
    config = xml_to_allometry.XmlConfig(altitude=150.0, ddy=1200.0, end_year=10)

    stand_data = xml_to_allometry.process_stand(
        config, parsed_stand, PEAT=1, output_dir=tmp_path
    )

    output_path = xml_to_allometry.plan_stand_output(parsed_stand, tmp_path)
    assert output_path.exists()
    df = read_allometry_info_from_csv(output_path)
    assert len(df) > 0

    # Only a dominant layer is ever produced by this tool.
    dominant_file = stand_data.allometry_file_per_layer[CanopyLayerName.dominant]
    assert dominant_file.file_path == output_path
    assert dominant_file.species_id == parsed_stand.main_species
    assert CanopyLayerName.subdominant not in stand_data.allometry_file_per_layer


def test_process_stand_returns_stand_data_with_the_shared_metadata_fields(tmp_path):
    parsed_stand = _parsed_stand_data("1")
    config = xml_to_allometry.XmlConfig(altitude=150.0, ddy=1200.0, end_year=10)

    stand_data = xml_to_allometry.process_stand(
        config, parsed_stand, PEAT=1, output_dir=tmp_path
    )

    assert stand_data.site_fertility_class == parsed_stand.fertility_class
    assert stand_data.x_ykj == parsed_stand.x_ykj
    assert stand_data.y_ykj == parsed_stand.y_ykj
    assert stand_data.polygon is not None
    assert stand_data.polygon.equals(parsed_stand.polygon)
    assert stand_data.stand_area == parsed_stand.area
    assert stand_data.soil_type == parsed_stand.soil_type
    # The one pooled curve sits on the dominant layer, so that's the only age.
    # This fixture has a single stratum (pine, age 40), so the pooled age is
    # just that stratum's age.
    assert stand_data.initial_age_per_layer == {CanopyLayerName.dominant: 40.0}


def test_process_stand_records_the_curves_pooled_start_age_not_tss_mean_age(tmp_path):
    # The two-species fixture stand: pine age 40 (G 20.0) and spruce age 35
    # (G 8.0), with tss:MeanAge 40. The growth model pools the strata's ages
    # weighted by basal area -- (40*20 + 35*8) / 28 = 38.57 -- and rounds, so
    # the curve starts at 39. The recorded initial age must be the curve's
    # start age, not the inventory's tss:MeanAge.
    parsed_stand = _parsed_stand_data("1", include_second_species=True)
    config = xml_to_allometry.XmlConfig(altitude=150.0, ddy=1200.0, end_year=10)

    stand_data = xml_to_allometry.process_stand(
        config, parsed_stand, PEAT=1, output_dir=tmp_path
    )

    assert stand_data.initial_age_per_layer == {CanopyLayerName.dominant: 39.0}


def test_process_stand_returns_raw_per_species_basal_areas_and_stem_counts(tmp_path):
    # The two-species fixture stand: a pine stratum (species 1, G 20.0 /
    # N 500) and a spruce one (species 2, G 8.0 / N 300). Nothing with a
    # species code >= 3, so the "deciduous" slot stays at ZERO_STRATUM.
    parsed_stand = _parsed_stand_data("1", include_second_species=True)
    config = xml_to_allometry.XmlConfig(altitude=150.0, ddy=1200.0, end_year=10)

    stand_data = xml_to_allometry.process_stand(
        config, parsed_stand, PEAT=1, output_dir=tmp_path
    )

    # Raw XML figures, not adjusted by any thinning heuristic.
    assert stand_data.basal_area_pine == 20.0
    assert stand_data.basal_area_spruce == 8.0
    assert stand_data.stem_count_pine == 500
    assert stand_data.stem_count_spruce == 300

    # A species with no stratum is the ZERO_STRATUM sentinel, so it records a
    # real 0.0 ("measured, none there") rather than None ("not recorded").
    assert stand_data.basal_area_deciduous == 0.0
    assert stand_data.stem_count_deciduous == 0.0


def test_dump_stand_data_document_is_the_shared_function():
    # dump_stand_data_document now lives in tools.shared_allometry_tool_utils.
    # stand_data, imported here instead of a local copy -- its write-to-disk
    # behavior is tested once, in test_shared_allometry_tool_utils.py.
    assert xml_to_allometry.dump_stand_data_document is shared_dump_stand_data_document


def test_print_dry_run_plan_reports_counts_and_writes_nothing(tmp_path, capsys):
    project_dir = tmp_path
    output_dir = project_dir / "inputs" / "allometry"
    parsed_stands = [_parsed_stand_data("1"), _parsed_stand_data("2")]

    xml_to_allometry.print_dry_run_plan(parsed_stands, output_dir, project_dir)

    printed = capsys.readouterr().out
    assert "2 stand(s) -- 2 CSV(s)" in printed
    # The JSON is reported beside the allometry folder in inputs/, not
    # inside it.
    assert str(project_dir / "inputs" / "stand_data.json") in printed
    assert str(output_dir / "stand_data.json") not in printed
    assert not output_dir.exists()


# %% parse_CLI_arguments


@pytest.fixture
def dummy_xml_file(tmp_path):
    xml_path = tmp_path / "stand.xml"
    xml_path.write_text(
        _forest_property_xml(_stand_xml_block("1", include_tree_strata=True))
    )
    return xml_path


@pytest.fixture
def project_dir(tmp_path):
    # --project-dir is the project root, and must already exist (shared
    # valid_existing_directory). Everything this tool reads and writes lives
    # in its inputs/ folder: config.toml going in, the allometry CSVs and
    # stand_data.json coming out.
    path = tmp_path / "myproject"
    (path / "inputs").mkdir(parents=True)
    return path


@pytest.fixture
def dummy_config_file(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text("altitude = 150.0\nddy = 1200.0\n")
    return config_path


def _run_parse_CLI_arguments(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["xml_to_allometry.py", *argv])
    return xml_to_allometry.parse_CLI_arguments()


def test_parse_CLI_arguments_requires_project_dir(
    monkeypatch, dummy_xml_file, dummy_config_file, capsys
):
    with pytest.raises(SystemExit):
        _run_parse_CLI_arguments(
            monkeypatch, [str(dummy_xml_file), f"--config={dummy_config_file}"]
        )
    assert "--project-dir" in capsys.readouterr().err


def test_parse_CLI_arguments_reports_a_missing_default_config(
    monkeypatch, dummy_xml_file, project_dir, capsys
):
    with pytest.raises(SystemExit):
        _run_parse_CLI_arguments(
            monkeypatch, [str(dummy_xml_file), f"--project-dir={project_dir}"]
        )
    stderr = capsys.readouterr().err
    assert str(project_dir / "inputs" / "config.toml") in stderr
    assert "--config" in stderr


def test_parse_CLI_arguments_finds_config_toml_inside_project_dir_by_default(
    monkeypatch, dummy_xml_file, project_dir
):
    (project_dir / "inputs" / "config.toml").write_text(
        "altitude = 150.0\nddy = 1200.0\n"
    )
    cli_args = _run_parse_CLI_arguments(
        monkeypatch, [str(dummy_xml_file), f"--project-dir={project_dir}"]
    )
    assert cli_args.config_path == project_dir / "inputs" / "config.toml"
    assert cli_args.config.altitude == 150.0


def test_parse_CLI_arguments_explicit_config_overrides_the_default_lookup(
    monkeypatch, dummy_xml_file, dummy_config_file, project_dir
):
    (project_dir / "inputs" / "config.toml").write_text(
        "altitude = 10.0\nddy = 600.0\n"
    )
    cli_args = _run_parse_CLI_arguments(
        monkeypatch,
        [
            str(dummy_xml_file),
            f"--config={dummy_config_file}",
            f"--project-dir={project_dir}",
        ],
    )
    assert cli_args.config_path == dummy_config_file
    assert (
        cli_args.config.altitude == 150.0
    )  # from dummy_config_file, not project_dir's own


def test_parse_CLI_arguments_blocks_out_of_range_altitude_by_default(
    monkeypatch, dummy_xml_file, project_dir, tmp_path, capsys
):
    config_path = tmp_path / "config.toml"
    config_path.write_text("altitude = 1500.0\nddy = 1200.0\n")
    with pytest.raises(SystemExit):
        _run_parse_CLI_arguments(
            monkeypatch,
            [
                str(dummy_xml_file),
                f"--config={config_path}",
                f"--project-dir={project_dir}",
            ],
        )
    stderr = capsys.readouterr().err
    assert "altitude" in stderr
    assert "--allow-out-of-range-values" in stderr


def test_parse_CLI_arguments_allows_out_of_range_with_override(
    monkeypatch, dummy_xml_file, project_dir, tmp_path, capsys
):
    config_path = tmp_path / "config.toml"
    config_path.write_text("altitude = 1500.0\nddy = 1200.0\n")
    cli_args = _run_parse_CLI_arguments(
        monkeypatch,
        [
            str(dummy_xml_file),
            f"--config={config_path}",
            f"--project-dir={project_dir}",
            "--allow-out-of-range-values",
        ],
    )
    assert cli_args.config.altitude == 1500.0
    assert "Warning" in capsys.readouterr().out


def test_parse_CLI_arguments_rejects_nan_even_with_override(
    monkeypatch, dummy_xml_file, project_dir, tmp_path, capsys
):
    config_path = tmp_path / "config.toml"
    config_path.write_text("altitude = nan\nddy = 1200.0\n")
    with pytest.raises(SystemExit):
        _run_parse_CLI_arguments(
            monkeypatch,
            [
                str(dummy_xml_file),
                f"--config={config_path}",
                f"--project-dir={project_dir}",
                "--allow-out-of-range-values",
            ],
        )
    assert "altitude" in capsys.readouterr().err


def test_parse_CLI_arguments_refuses_existing_default_output_dir(
    monkeypatch, dummy_xml_file, dummy_config_file, project_dir, capsys
):
    (project_dir / "inputs" / "allometry").mkdir()
    with pytest.raises(SystemExit):
        _run_parse_CLI_arguments(
            monkeypatch,
            [
                str(dummy_xml_file),
                f"--config={dummy_config_file}",
                f"--project-dir={project_dir}",
            ],
        )
    stderr = capsys.readouterr().err
    assert "already exists" in stderr
    assert "--project-dir" in stderr


def test_parse_CLI_arguments_creates_no_output_folder(
    monkeypatch, dummy_xml_file, dummy_config_file, project_dir
):
    _run_parse_CLI_arguments(
        monkeypatch,
        [
            str(dummy_xml_file),
            f"--config={dummy_config_file}",
            f"--project-dir={project_dir}",
        ],
    )
    assert not (project_dir / "inputs" / "allometry").exists()


def test_parse_CLI_arguments_dry_run_defaults_to_false(
    monkeypatch, dummy_xml_file, dummy_config_file, project_dir
):
    cli_args = _run_parse_CLI_arguments(
        monkeypatch,
        [
            str(dummy_xml_file),
            f"--config={dummy_config_file}",
            f"--project-dir={project_dir}",
        ],
    )
    assert cli_args.dry_run is False


def test_parse_CLI_arguments_dry_run_creates_no_output_folder(
    monkeypatch, dummy_xml_file, dummy_config_file, project_dir
):
    cli_args = _run_parse_CLI_arguments(
        monkeypatch,
        [
            str(dummy_xml_file),
            f"--config={dummy_config_file}",
            f"--project-dir={project_dir}",
            "--dry-run",
        ],
    )
    assert cli_args.dry_run is True
    assert not (project_dir / "inputs" / "allometry").exists()


def test_parse_CLI_arguments_dry_run_still_refuses_existing_output_dir(
    monkeypatch, dummy_xml_file, dummy_config_file, project_dir, capsys
):
    (project_dir / "inputs" / "allometry").mkdir()
    with pytest.raises(SystemExit):
        _run_parse_CLI_arguments(
            monkeypatch,
            [
                str(dummy_xml_file),
                f"--config={dummy_config_file}",
                f"--project-dir={project_dir}",
                "--dry-run",
            ],
        )
    assert "already exists" in capsys.readouterr().err


# %% main


def test_main_writes_crs_and_wkt_polygons_into_stand_data_json(
    monkeypatch, tmp_path, project_dir
):
    xml_path = tmp_path / "stands.xml"
    xml_path.write_text(
        _forest_property_xml(
            _stand_xml_block(
                "1",
                include_tree_strata=True,
                interior_coordinates=(_hole_coordinates(),),
            )
        )
    )
    (project_dir / "inputs" / "config.toml").write_text(
        "altitude = 150.0\nddy = 1200.0\nend_year = 10\n"
    )
    monkeypatch.setattr(
        "sys.argv",
        ["xml_to_allometry.py", str(xml_path), f"--project-dir={project_dir}"],
    )

    xml_to_allometry.main()

    json_path = project_dir / "inputs" / "stand_data.json"
    assert '"polygon":"POLYGON ((' in json_path.read_text()
    document = load_stand_data_document_from_json(json_path)
    assert document.crs == SOURCE_CRS
    polygon = document.stands[StandID("1")].polygon
    assert polygon is not None
    assert len(polygon.interiors) == 1


def test_main_records_allometry_paths_relative_to_stand_data_json(
    monkeypatch, tmp_path, project_dir
):
    # Run from project_dir's parent with a relative --project-dir: the
    # recorded path must not depend on either (docs/adr/0005).
    xml_path = tmp_path / "stands.xml"
    xml_path.write_text(
        _forest_property_xml(_stand_xml_block("1", include_tree_strata=True))
    )
    (project_dir / "inputs" / "config.toml").write_text(
        "altitude = 150.0\nddy = 1200.0\nend_year = 10\n"
    )
    monkeypatch.chdir(project_dir.parent)
    monkeypatch.setattr(
        "sys.argv",
        ["xml_to_allometry.py", str(xml_path), f"--project-dir={project_dir.name}"],
    )

    xml_to_allometry.main()

    json_path = project_dir / "inputs" / "stand_data.json"
    raw = json.loads(json_path.read_text())
    assert raw["stands"]["1"]["allometry_file_per_layer"]["dominant"]["file_path"] == (
        "allometry/1.csv"
    )
    dominant = (
        load_stand_data_document_from_json(json_path)
        .stands[StandID("1")]
        .allometry_file_per_layer[CanopyLayerName.dominant]
    )
    assert dominant.file_path == project_dir / "inputs" / "allometry" / "1.csv"
    assert dominant.file_path.exists()

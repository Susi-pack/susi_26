import json
import math
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from shapely.geometry import MultiPolygon, Point, Polygon

from susi.io.load_output_data import StandID
from susi.io.utils import SRC_DIR
from susi.io.susi_parameter_model import CanopyLayerName, read_allometry_info_from_csv
from tools.metsakeskus_to_allometry import metsakeskus_to_allometry as m
from tools.shared_allometry_tool_utils import input_validation, shared_utils
from tools.shared_allometry_tool_utils.stand_data import (
    StandDataDocument,
    dump_stand_data_document as shared_dump_stand_data_document,
    load_stand_data_document_from_json,
)

# %% ExtractionConfig
#
# ExtractionConfig is a StrictFrozenModel (susi.io.extra_pydantic_types):
# presence/unknown-field checking, required-vs-defaulted fields, and
# extra="forbid" all come from Pydantic itself, the same as
# new_growth_allometry.py's NewGrowthConfig.


def test_extraction_config_requires_target_year():
    with pytest.raises(ValueError, match="target_year"):
        m.ExtractionConfig.model_validate({"altitude": 150.0, "ddy": 1200.0})


def test_extraction_config_requires_altitude():
    with pytest.raises(ValueError, match="altitude"):
        m.ExtractionConfig.model_validate({"target_year": 2018, "ddy": 1200.0})


def test_extraction_config_requires_ddy():
    with pytest.raises(ValueError, match="ddy"):
        m.ExtractionConfig.model_validate({"target_year": 2018, "altitude": 150.0})


def test_extraction_config_reports_all_missing_fields_together():
    with pytest.raises(ValueError) as exc_info:
        m.ExtractionConfig.model_validate({})
    message = str(exc_info.value)
    assert "target_year" in message
    assert "altitude" in message
    assert "ddy" in message


def test_extraction_config_rejects_unknown_field():
    with pytest.raises(ValueError, match="typo_field"):
        m.ExtractionConfig.model_validate(
            {"target_year": 2018, "altitude": 150.0, "ddy": 1200.0, "typo_field": 1}
        )


def test_extraction_config_applies_defaults():
    config = m.ExtractionConfig.model_validate(
        {"target_year": 2018, "altitude": 150.0, "ddy": 1200.0}
    )
    assert config.developmentclass_filter == (1, 2, 3)
    assert config.fertilityclass_filter == (2, 3, 4, 5)
    assert config.n_trees == 20
    assert config.start_year == 5
    assert config.end_year == 80
    assert config.step_years == 5


def test_extraction_config_overrides_defaults():
    config = m.ExtractionConfig.model_validate(
        {
            "target_year": 2018,
            "altitude": 150.0,
            "ddy": 1200.0,
            "developmentclass_filter": [2, 3],
            "n_trees": 5,
        }
    )
    assert config.developmentclass_filter == (2, 3)
    assert config.n_trees == 5


def test_extraction_config_is_frozen():
    config = m.ExtractionConfig(target_year=2018, altitude=150.0, ddy=1200.0)
    with pytest.raises(Exception):  # noqa: B017 -- pydantic's frozen-model error
        setattr(config, "target_year", 2019)  # noqa: B010 -- setattr, not `.` access, to keep this a runtime-only check


def test_load_extraction_config_reads_toml(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "target_year = 2018\naltitude = 150.0\nddy = 1200.0\nn_trees = 15\n"
    )
    config = m.load_extraction_config(config_path)
    assert config.target_year == 2018
    assert config.n_trees == 15


# %% default_config.toml
#
# Guards against the shipped default/template config drifting from
# ExtractionConfig's own field defaults -- see that file's header comment.

DEFAULT_CONFIG_PATH = (
    SRC_DIR / "tools" / "metsakeskus_to_allometry" / "default_config.toml"
)


def test_default_config_toml_optional_fields_match_model_defaults():
    config = m.load_extraction_config(DEFAULT_CONFIG_PATH)
    defaults = m.ExtractionConfig(target_year=0, altitude=0.0, ddy=0.0)
    assert config.developmentclass_filter == defaults.developmentclass_filter
    assert config.fertilityclass_filter == defaults.fertilityclass_filter
    assert config.n_trees == defaults.n_trees
    assert config.start_year == defaults.start_year
    assert config.end_year == defaults.end_year
    assert config.step_years == defaults.step_years


def test_default_config_toml_required_fields_are_deliberately_out_of_range():
    """The shipped file's altitude/ddy placeholders must stay outside
    [ALTITUDE_MIN, ALTITUDE_MAX] / [DDY_MIN, DDY_MAX] -- that's what makes an
    untouched copy fail loudly instead of running silently (see the file's
    header comment and parse_CLI_arguments' out-of-range handling)."""
    config = m.load_extraction_config(DEFAULT_CONFIG_PATH)
    assert not (
        input_validation.ALTITUDE_MIN
        <= config.altitude
        <= input_validation.ALTITUDE_MAX
    )
    assert not (input_validation.DDY_MIN <= config.ddy <= input_validation.DDY_MAX)


# %% out_of_range_message reuse (from the shared package)


def test_out_of_range_message_reused_from_shared_package():
    assert input_validation.out_of_range_message("altitude", 500, 0, 1000) is None
    message = input_validation.out_of_range_message("altitude", -1, 0, 1000)
    assert message is not None
    assert "altitude" in message


# %% filter_stands_by_site_attributes


def _stand_row(standid, maingroup=1, subgroup=2, drainagestate=7, fertilityclass=3):
    return {
        "standid": standid,
        "maingroup": maingroup,
        "subgroup": subgroup,
        "drainagestate": drainagestate,
        "fertilityclass": fertilityclass,
        "developmentclass": 2,
        "soiltype": 10,
        "area": 1.5,
        "geometry": Point(0, 0).buffer(1),
    }


def _make_stand_gdf(rows):
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:3067")


def test_filter_stands_by_site_attributes_keeps_matching_stand():
    stand = _make_stand_gdf([_stand_row(1)])
    result = m.filter_stands_by_site_attributes(
        stand, fertilityclass_filter=(2, 3, 4, 5)
    )
    assert list(result["standid"]) == [1]


@pytest.mark.parametrize(
    "override",
    [
        {"maingroup": 2},  # not forest land
        {"subgroup": 1},  # not peatland (Kangas, mineral soil)
        {"drainagestate": 6},  # undrained
        {"fertilityclass": 1},  # outside configured range
    ],
)
def test_filter_stands_by_site_attributes_excludes_non_matching_stand(override):
    stand = _make_stand_gdf([_stand_row(1, **override)])
    result = m.filter_stands_by_site_attributes(
        stand, fertilityclass_filter=(2, 3, 4, 5)
    )
    assert len(result) == 0


def test_filter_stands_by_site_attributes_respects_configured_fertilityclass_range():
    stand = _make_stand_gdf([_stand_row(1, fertilityclass=6)])
    result = m.filter_stands_by_site_attributes(
        stand, fertilityclass_filter=(2, 3, 4, 5, 6)
    )
    assert list(result["standid"]) == [1]


# %% select_target_year_snapshot


def _treestand_row(standid, treestandid, date, type_, standid_col="standid"):
    return {"standid": standid, "treestandid": treestandid, "date": date, "type": type_}


def test_select_target_year_snapshot_keeps_exact_year_match():
    treestand = pd.DataFrame(
        [
            _treestand_row(1, 101, "2018-05-01", 1),
            _treestand_row(1, 102, "2026-01-01", 2),  # projected snapshot, not measured
        ]
    )
    result = m.select_target_year_snapshot(treestand, {1}, target_year=2018)
    assert list(result["treestandid"]) == [101]


def test_select_target_year_snapshot_excludes_stand_without_exact_year():
    treestand = pd.DataFrame([_treestand_row(1, 101, "2015-05-01", 1)])
    result = m.select_target_year_snapshot(treestand, {1}, target_year=2018)
    assert (
        len(result) == 0
    )  # no ">= target_year, else next" fallback -- see module docstring


def test_select_target_year_snapshot_ignores_unmeasured_types():
    treestand = pd.DataFrame([_treestand_row(1, 101, "2018-05-01", 2)])
    result = m.select_target_year_snapshot(treestand, {1}, target_year=2018)
    assert len(result) == 0


def test_select_target_year_snapshot_ignores_other_stands():
    treestand = pd.DataFrame([_treestand_row(2, 201, "2018-05-01", 1)])
    result = m.select_target_year_snapshot(treestand, {1}, target_year=2018)
    assert len(result) == 0


def test_select_target_year_snapshot_keeps_one_row_per_stand():
    treestand = pd.DataFrame(
        [
            _treestand_row(1, 101, "2018-01-01", 1),
            _treestand_row(1, 102, "2018-06-01", 1),
        ]
    )
    result = m.select_target_year_snapshot(treestand, {1}, target_year=2018)
    assert len(result) == 1


# %% filter_by_developmentclass


def test_filter_by_developmentclass_keeps_configured_classes():
    merged = pd.DataFrame({"standid": [1, 2, 3], "developmentclass": [1, 4, 2]})
    result = m.filter_by_developmentclass(merged, developmentclass_filter=(1, 2, 3))
    assert sorted(result["standid"]) == [1, 3]


# %% basal_area_per_tree_m2 / estimate_stemcount


def test_basal_area_per_tree_m2_known_value():
    # A 20cm-diameter tree: BA = pi * (0.1m)^2 = 0.0314159... m2
    assert m.basal_area_per_tree_m2(20) == pytest.approx(math.pi * 0.1**2)


def test_estimate_stemcount_backs_out_a_plausible_count():
    # 20 m2/ha spread over 20cm-diameter trees (~0.0314 m2/tree) -> ~637 stems/ha
    assert m.estimate_stemcount(20.0, 20.0) == pytest.approx(637, abs=1)


def test_estimate_stemcount_zero_basal_area_returns_zero():
    assert m.estimate_stemcount(0.0, 20.0) == 0


def test_estimate_stemcount_zero_diameter_returns_zero():
    assert m.estimate_stemcount(20.0, 0.0) == 0


# %% aggregate_species_group


def test_aggregate_species_group_empty_returns_zero_stratum():
    # No treestratum rows at all for this species -- nothing recorded to
    # preserve, so this is literally ZERO_STRATUM (see its docstring).
    empty = pd.DataFrame(
        {
            "age": [],
            "basalarea": [],
            "stemcount": [],
            "meandiameter": [],
            "meanheight": [],
        }
    )
    result = m.aggregate_species_group(empty, species_name="pine")
    assert result is m.ZERO_STRATUM
    assert result == m.TreeStratum(
        age=0, basal_area=0.0, stem_count=0, mean_diameter=0.0, mean_height=0.0
    )


def test_aggregate_species_group_weights_by_basal_area():
    rows = pd.DataFrame(
        {
            "age": [20.0, 40.0],
            "basalarea": [10.0, 30.0],
            "stemcount": [100.0, 300.0],
            "meandiameter": [10.0, 20.0],
            "meanheight": [8.0, 16.0],
        }
    )
    result = m.aggregate_species_group(rows, species_name="pine")
    # Weighted age: (20*10 + 40*30) / 40 = 35
    assert result.age == 35
    assert result.basal_area == pytest.approx(40.0)
    assert result.stem_count == 400
    assert result.mean_diameter == pytest.approx(17.5)
    assert result.mean_height == pytest.approx(14.0)


def test_aggregate_species_group_estimates_missing_stemcount():
    rows = pd.DataFrame(
        {
            "age": [30.0],
            "basalarea": [20.0],
            "stemcount": [0.0],
            "meandiameter": [20.0],
            "meanheight": [15.0],
        }
    )
    result = m.aggregate_species_group(rows, species_name="pine")
    assert result.stem_count > 0


def test_aggregate_species_group_raises_on_degenerate_diameter_with_real_basal_area():
    # Real, positive basal area but no usable mean diameter: this species
    # would reach Growth_and_Yield_Table if it becomes dominant/subdominant,
    # so there is no safe value to invent -- reject rather than paper over it
    # (previously substituted a fabricated NOMINAL_DIAMETER_CM here).
    rows = pd.DataFrame(
        {
            "age": [30.0],
            "basalarea": [20.0],
            "stemcount": [100.0],
            "meandiameter": [0.0],
            "meanheight": [15.0],
        }
    )
    with pytest.raises(m.DegenerateSpeciesDataError, match="deciduous"):
        m.aggregate_species_group(rows, species_name="deciduous")


def test_aggregate_species_group_raises_on_degenerate_height_with_real_basal_area():
    rows = pd.DataFrame(
        {
            "age": [30.0],
            "basalarea": [20.0],
            "stemcount": [100.0],
            "meandiameter": [20.0],
            "meanheight": [0.0],
        }
    )
    with pytest.raises(m.DegenerateSpeciesDataError, match="spruce"):
        m.aggregate_species_group(rows, species_name="spruce")


def test_aggregate_species_group_healthy_basal_area_does_not_raise():
    rows = pd.DataFrame(
        {
            "age": [30.0],
            "basalarea": [20.0],
            "stemcount": [100.0],
            "meandiameter": [20.0],
            "meanheight": [15.0],
        }
    )
    result = m.aggregate_species_group(rows, species_name="pine")
    assert result.mean_diameter == pytest.approx(20.0)
    assert result.mean_height == pytest.approx(15.0)


def test_aggregate_species_group_zero_basal_area_preserves_recorded_data():
    # The species IS recorded (rows are not empty) but its rows sum to zero
    # basal area -- e.g. a young regeneration cohort recorded without a
    # basal-area figure. Its real stem count, age, diameter and height must
    # be kept, not rewritten to zero just because basal_area is zero.
    rows = pd.DataFrame(
        {
            "age": [12.0],
            "basalarea": [0.0],
            "stemcount": [80.0],
            "meandiameter": [3.5],
            "meanheight": [2.1],
        }
    )
    result = m.aggregate_species_group(rows, species_name="spruce")
    assert result.basal_area == 0.0
    assert result.stem_count == 80
    assert result.age == 12
    assert result.mean_diameter == pytest.approx(3.5)
    assert result.mean_height == pytest.approx(2.1)


def test_aggregate_species_group_zero_basal_area_averages_multiple_rows():
    rows = pd.DataFrame(
        {
            "age": [10.0, 20.0],
            "basalarea": [0.0, 0.0],
            "stemcount": [50.0, 30.0],
            "meandiameter": [2.0, 4.0],
            "meanheight": [1.0, 3.0],
        }
    )
    result = m.aggregate_species_group(rows, species_name="deciduous")
    assert result.basal_area == 0.0
    assert result.stem_count == 80  # real stem count is summed, not zeroed
    assert (
        result.age == 15
    )  # plain average, not basal-area-weighted (no BA to weight by)
    assert result.mean_diameter == pytest.approx(3.0)
    assert result.mean_height == pytest.approx(2.0)


def test_aggregate_species_group_zero_basal_area_with_nothing_recorded_falls_back_to_zero_not_nan():
    # Rows are present but every other field is genuinely missing (NaN).
    # There is truly nothing to average, so this falls back to 0 -- never
    # NaN, since NaN * 0 == NaN would silently poison
    # build_stand's basal-area-weighted stand-level age/height/
    # diameter for this species' basal_area=0 weight.
    rows = pd.DataFrame(
        {
            "age": [None],
            "basalarea": [0.0],
            "stemcount": [0.0],
            "meandiameter": [None],
            "meanheight": [None],
        }
    )
    result = m.aggregate_species_group(rows, species_name="deciduous")
    assert result.age == 0
    assert result.stem_count == 0
    assert result.mean_diameter == 0.0
    assert result.mean_height == 0.0
    assert not math.isnan(result.mean_diameter)
    assert not math.isnan(result.mean_height)


# %% build_species_strata


def test_build_species_strata_propagates_degenerate_species_data_error():
    stand_strata = pd.DataFrame(
        {
            "treespecies": [1, 2],
            "age": [45.0, 40.0],
            "basalarea": [10.0, 5.0],
            "stemcount": [0.0, 0.0],
            "meandiameter": [20.0, 0.0],  # spruce: real BA, degenerate diameter
            "meanheight": [18.0, 0.0],
        }
    )
    with pytest.raises(m.DegenerateSpeciesDataError, match="spruce"):
        m.build_species_strata(stand_strata)


# %% total_basal_area / determine_dominant_and_subdominant_species


def _strata(pine_ba, spruce_ba, decid_ba):
    return m.PerSpecies(
        pine=m.TreeStratum(
            age=30, basal_area=pine_ba, stem_count=100, mean_diameter=15, mean_height=12
        ),
        spruce=m.TreeStratum(
            age=30,
            basal_area=spruce_ba,
            stem_count=100,
            mean_diameter=15,
            mean_height=12,
        ),
        deciduous=m.TreeStratum(
            age=30,
            basal_area=decid_ba,
            stem_count=100,
            mean_diameter=15,
            mean_height=12,
        ),
    )


def test_total_basal_area_sums_all_three_species():
    assert m.total_basal_area(_strata(10, 5, 2)) == pytest.approx(17)


def test_determine_dominant_and_subdominant_species_ranks_by_basal_area():
    dominant, subdominant = m.determine_dominant_and_subdominant_species(
        _strata(10, 20, 5)
    )
    assert dominant == 2  # spruce
    assert subdominant == 1  # pine


def test_determine_dominant_and_subdominant_species_monoculture_keeps_zero_ba_subdominant():
    # Pure pine stand: spruce and deciduous both carry zero basal area.
    # Per docs/adr/0002, the runner-up (whichever it is) is still reported
    # as the subdominant -- never duplicated from the dominant.
    dominant, subdominant = m.determine_dominant_and_subdominant_species(
        _strata(30, 0, 0)
    )
    assert dominant == 1
    assert subdominant in (2, 3)


# %% isolate_species_layer


def test_isolate_species_layer_zeroes_other_species():
    strata = _strata(10, 20, 5)
    layer = m.isolate_species_layer(strata, active_species=2)
    assert layer.spruce == strata.spruce
    assert layer.pine.basal_area == 0.0
    assert layer.deciduous.basal_area == 0.0


# %% centroid_to_ykj
#
# Now shared_utils.centroid_to_ykj, tested in
# test_shared_allometry_tool_utils.py -- only the "this tool uses the shared
# function" check stays here.


def test_centroid_to_ykj_is_the_shared_function():
    assert m.centroid_to_ykj is shared_utils.centroid_to_ykj


# %% build_stand_candidates


_DEFAULT_GEOMETRY = Point(385000, 6685000).buffer(50)
_UNSET = object()


def _merged_row(standid=1, treestandid=101, geometry=_UNSET, soiltype=10, area=0.8):
    return pd.Series(
        {
            "area": area,
            "standid": standid,
            "treestandid": treestandid,
            "subgroup": 2,
            "fertilityclass": 3,
            "developmentclass": 2,
            "drainagestate": 7,
            "soiltype": soiltype,
            "geometry": _DEFAULT_GEOMETRY if geometry is _UNSET else geometry,
        }
    )


def _empty_treestratum():
    return pd.DataFrame(
        {
            "treestandid": [],
            "treespecies": [],
            "age": [],
            "basalarea": [],
            "stemcount": [],
            "meandiameter": [],
            "meanheight": [],
        }
    )


def test_build_stand_candidates_skips_none_geometry():
    merged_filtered = pd.DataFrame([_merged_row(geometry=None)])
    candidates, skipped = m.build_stand_candidates(
        merged_filtered, _empty_treestratum()
    )
    assert len(candidates) == 0
    assert len(skipped) == 1
    assert "geometry" in skipped[0].reason


def test_build_stand_candidates_skips_nan_geometry():
    # A NaN geometry cell (e.g. from an unmatched merge key) is float NaN,
    # not None -- must be caught the same way.
    merged_filtered = pd.DataFrame([_merged_row(geometry=float("nan"))])
    candidates, skipped = m.build_stand_candidates(
        merged_filtered, _empty_treestratum()
    )
    assert len(candidates) == 0
    assert len(skipped) == 1


def test_build_stand_candidates_skips_empty_geometry():
    merged_filtered = pd.DataFrame([_merged_row(geometry=Polygon())])
    candidates, skipped = m.build_stand_candidates(
        merged_filtered, _empty_treestratum()
    )
    assert len(candidates) == 0
    assert len(skipped) == 1


def test_build_stand_candidates_skips_a_non_polygon_geometry():
    # The real stand layer is all single Polygons, but a MultiPolygon (or any
    # other type) must become a clean skip here, not a StandData validation
    # failure later on.
    multipolygon = MultiPolygon([_DEFAULT_GEOMETRY])
    merged_filtered = pd.DataFrame([_merged_row(geometry=multipolygon)])
    candidates, skipped = m.build_stand_candidates(
        merged_filtered, _empty_treestratum()
    )
    assert len(candidates) == 0
    assert len(skipped) == 1
    assert "MultiPolygon" in skipped[0].reason


def test_build_stand_candidates_carries_the_area_column():
    merged_filtered = pd.DataFrame([_merged_row(area=0.8)])
    candidates, _ = m.build_stand_candidates(merged_filtered, _empty_treestratum())
    assert candidates[0].area == 0.8


def test_build_stand_candidates_missing_area_stays_none():
    # "Not recorded", same as a missing soiltype -- stand_areas.py reports
    # a stand with no stand_area by name rather than guessing one.
    merged_filtered = pd.DataFrame([_merged_row(area=float("nan"))])
    candidates, _ = m.build_stand_candidates(merged_filtered, _empty_treestratum())
    assert candidates[0].area is None


def test_build_stand_candidates_skips_missing_treestandid():
    merged_filtered = pd.DataFrame([_merged_row(treestandid=None)])
    candidates, skipped = m.build_stand_candidates(
        merged_filtered, _empty_treestratum()
    )
    assert len(candidates) == 0
    assert len(skipped) == 1
    assert "treestandid" in skipped[0].reason


def test_build_stand_candidates_keeps_valid_row():
    merged_filtered = pd.DataFrame([_merged_row()])
    candidates, skipped = m.build_stand_candidates(
        merged_filtered, _empty_treestratum()
    )
    assert len(candidates) == 1
    assert len(skipped) == 0


def test_build_stand_candidates_skips_degenerate_species_data():
    # treestandid 101 matches _merged_row()'s default -- a species with
    # real, positive basal area but no usable diameter (DegenerateSpeciesDataError,
    # see aggregate_species_group) must turn the whole stand into a
    # StandSkipped, not a candidate carrying fabricated allometry inputs.
    merged_filtered = pd.DataFrame([_merged_row()])
    treestratum = pd.DataFrame(
        [
            {
                "treestandid": 101,
                "treespecies": 1,
                "age": 45.0,
                "basalarea": 10.0,
                "stemcount": 0.0,
                "meandiameter": 0.0,
                "meanheight": 0.0,
            }
        ]
    )
    candidates, skipped = m.build_stand_candidates(merged_filtered, treestratum)
    assert len(candidates) == 0
    assert len(skipped) == 1
    assert "pine" in skipped[0].reason
    assert "basal area" in skipped[0].reason


def test_build_stand_candidates_preserves_zero_basal_area_species_data():
    # A species recorded with real stem count/diameter/height but zero
    # basal area (e.g. an unmeasured regeneration cohort) must survive into
    # the StandCandidate unchanged -- it is only ever excluded from the
    # allometry-creation step downstream, never from the data itself.
    merged_filtered = pd.DataFrame([_merged_row()])
    treestratum = pd.DataFrame(
        [
            {
                "treestandid": 101,
                "treespecies": 1,
                "age": 45.0,
                "basalarea": 15.0,
                "stemcount": 400.0,
                "meandiameter": 20.0,
                "meanheight": 18.0,
            },
            {
                "treestandid": 101,
                "treespecies": 2,
                "age": 12.0,
                "basalarea": 0.0,
                "stemcount": 80.0,
                "meandiameter": 3.5,
                "meanheight": 2.1,
            },
        ]
    )
    candidates, skipped = m.build_stand_candidates(merged_filtered, treestratum)
    assert len(skipped) == 0
    assert len(candidates) == 1
    spruce = candidates[0].strata.spruce
    assert spruce.basal_area == 0.0
    assert spruce.stem_count == 80
    assert spruce.age == 12
    assert spruce.mean_diameter == pytest.approx(3.5)
    assert spruce.mean_height == pytest.approx(2.1)


def test_build_stand_candidates_missing_soiltype_stays_none():
    # No soiltype recorded for this stand: must stay None, not a fabricated
    # placeholder number indistinguishable from a real measurement (soiltype
    # never feeds Growth_and_Yield_Table -- see StandCandidate).
    merged_filtered = pd.DataFrame([_merged_row(soiltype=None)])
    candidates, skipped = m.build_stand_candidates(
        merged_filtered, _empty_treestratum()
    )
    assert not skipped
    assert candidates[0].soiltype is None


def test_build_stand_candidates_keeps_recorded_soiltype():
    merged_filtered = pd.DataFrame([_merged_row(soiltype=10)])
    candidates, skipped = m.build_stand_candidates(
        merged_filtered, _empty_treestratum()
    )
    assert not skipped
    assert candidates[0].soiltype == 10


# %% partition_viable_candidates


def _candidate(stand_id, pine_ba, spruce_ba=0, decid_ba=0, soiltype=10, area=0.8):
    return m.StandCandidate(
        area=area,
        id=StandID(stand_id),
        subgroup=2,
        fertilityclass=3,
        developmentclass=2,
        drainagestate=7,
        soiltype=soiltype,
        strata=_strata(pine_ba, spruce_ba, decid_ba),
        # A real Helsinki-area point (EPSG:3067) rather than (0, 0): keeps
        # centroid_to_ykj's output inside Growth_and_Yield_Table's expected
        # Finnish coordinate range, avoiding spurious NaN/log-domain warnings
        # from its internal assortment/height model.
        geometry=Point(385000, 6685000).buffer(50),
    )


def test_partition_viable_candidates_keeps_nonzero_basal_area():
    viable, skipped = m.partition_viable_candidates([_candidate("1", pine_ba=10)])
    assert len(viable) == 1
    assert len(skipped) == 0


def test_partition_viable_candidates_skips_zero_basal_area():
    viable, skipped = m.partition_viable_candidates([_candidate("1", pine_ba=0)])
    assert len(viable) == 0
    assert len(skipped) == 1
    assert skipped[0].stand_id == StandID("1")
    assert "basal area" in skipped[0].reason


# %% build_stand


def test_build_stand_is_total_for_a_viable_candidate():
    candidate = _candidate("1", pine_ba=10, spruce_ba=5)
    stand = m.build_stand(candidate)
    assert stand.id == StandID("1")
    assert stand.stand_basalarea == pytest.approx(15)
    assert stand.dominant_species == 1
    assert stand.subdominant_species == 2


def test_build_stands_isolates_one_bad_candidate(monkeypatch):
    # build_stand can still raise for a structurally pathological
    # geometry (e.g. all-coincident points) even after partition_viable_candidates
    # -- one such candidate must not abort every other stand in the batch.
    good = _candidate("1", pine_ba=10)
    bad = _candidate("2", pine_ba=10)

    real_centroid_to_ykj = m.centroid_to_ykj

    def _flaky_centroid_to_ykj(geometry):
        if geometry is bad.geometry:
            raise ValueError("degenerate geometry")
        return real_centroid_to_ykj(geometry)

    monkeypatch.setattr(m, "centroid_to_ykj", _flaky_centroid_to_ykj)

    built, skipped = m.build_stands([good, bad])

    assert [s.id for s in built] == [StandID("1")]
    assert [s.stand_id for s in skipped] == [StandID("2")]
    assert "degenerate geometry" in skipped[0].reason


# %% Zero-basal-area species data survives into ParsedStand but never
# reaches allometry creation (end-to-end, through the real aggregation +
# growth-table pipeline rather than the hand-built _strata/_candidate
# helpers above, which bypass aggregate_species_group entirely).


def _stand_candidate_with_a_zero_basal_area_species(treestratum_rows):
    merged_filtered = pd.DataFrame([_merged_row()])
    candidates, skipped = m.build_stand_candidates(
        merged_filtered, pd.DataFrame(treestratum_rows)
    )
    assert not skipped
    return candidates[0]


def test_build_stand_ignores_preserved_zero_basal_area_species_in_stand_level_average():
    # Spruce carries real, non-nominal age/diameter/height despite zero
    # basal area -- it must contribute nothing (weight 0) to the stand-level
    # basal-area-weighted averages, which should equal pine's own values.
    candidate = _stand_candidate_with_a_zero_basal_area_species(
        [
            {
                "treestandid": 101,
                "treespecies": 1,
                "age": 45.0,
                "basalarea": 15.0,
                "stemcount": 400.0,
                "meandiameter": 20.0,
                "meanheight": 18.0,
            },
            {
                "treestandid": 101,
                "treespecies": 2,
                "age": 12.0,
                "basalarea": 0.0,
                "stemcount": 80.0,
                "meandiameter": 3.5,
                "meanheight": 2.1,
            },
        ]
    )
    viable, skipped = m.partition_viable_candidates([candidate])
    assert not skipped
    stand = m.build_stand(viable[0])

    assert stand.stand_meanage == pytest.approx(45.0)
    assert stand.stand_meandiameter == pytest.approx(20.0)
    assert stand.stand_meanheight == pytest.approx(18.0)


def test_process_stand_excludes_zero_basal_area_species_even_with_real_data(tmp_path):
    # Same setup, but check the actual allometry-creation step: spruce's
    # real recorded data must not turn into a subdominant CSV, because it
    # never has positive basal area (process_stand's own gate).
    candidate = _stand_candidate_with_a_zero_basal_area_species(
        [
            {
                "treestandid": 101,
                "treespecies": 1,
                "age": 45.0,
                "basalarea": 15.0,
                "stemcount": 400.0,
                "meandiameter": 20.0,
                "meanheight": 18.0,
            },
            {
                "treestandid": 101,
                "treespecies": 2,
                "age": 12.0,
                "basalarea": 0.0,
                "stemcount": 80.0,
                "meandiameter": 3.5,
                "meanheight": 2.1,
            },
        ]
    )
    viable, _ = m.partition_viable_candidates([candidate])
    stand = m.build_stand(viable[0])
    # Spruce's real data is still there, for reporting/future use ...
    assert stand.strata.spruce.stem_count == 80
    config = m.ExtractionConfig(
        target_year=2018, altitude=150.0, ddy=1200.0, end_year=10
    )

    outcome = m.process_stand(stand, config, tmp_path)

    # ... but it never became an allometry CSV.
    assert isinstance(outcome, m.StandWritten)
    assert outcome.subdominant_csv is None
    assert list(tmp_path.glob("*.csv")) == [outcome.dominant_csv]


# %% write_allometry_csv round-trips through the real SUSI reader


def test_write_allometry_csv_round_trips_through_read_allometry_info_from_csv(tmp_path):
    candidate = _candidate("1", pine_ba=10)
    stand = m.build_stand(candidate)
    config = m.ExtractionConfig(
        target_year=2018, altitude=150.0, ddy=1200.0, end_year=10
    )

    table = m.build_growth_and_yield_table(
        stand.strata,
        stand.dominant_species,
        stand.fertilityclass,
        stand.x_ykj,
        stand.y_ykj,
        config.altitude,
        config.ddy,
        config.n_trees,
        config.start_year,
        config.end_year,
        config.step_years,
    )
    output_path = tmp_path / "1_dominant.csv"
    m.write_allometry_csv(table, output_path)

    df = read_allometry_info_from_csv(output_path)
    assert len(df) > 0


# %% process_stand


def test_process_stand_writes_two_csvs(tmp_path):
    candidate = _candidate("1", pine_ba=10, spruce_ba=5)
    stand = m.build_stand(candidate)
    config = m.ExtractionConfig(
        target_year=2018, altitude=150.0, ddy=1200.0, end_year=10
    )

    outcome = m.process_stand(stand, config, tmp_path)

    assert isinstance(outcome, m.StandWritten)
    assert outcome.dominant_csv.exists()
    assert outcome.subdominant_csv is not None
    assert outcome.subdominant_csv.exists()


def test_process_stand_monoculture_writes_only_a_dominant_csv(tmp_path):
    # A pure pine stand: spruce and deciduous both carry zero basal area, so
    # whichever ranks second as "subdominant" has no live trees to model.
    candidate = _candidate("1", pine_ba=10)
    stand = m.build_stand(candidate)
    config = m.ExtractionConfig(
        target_year=2018, altitude=150.0, ddy=1200.0, end_year=10
    )

    outcome = m.process_stand(stand, config, tmp_path)

    assert isinstance(outcome, m.StandWritten)
    assert outcome.dominant_csv.exists()
    assert outcome.subdominant_csv is None
    # Only the dominant file actually landed on disk.
    assert list(tmp_path.glob("*.csv")) == [outcome.dominant_csv]


def test_process_stand_skips_on_failure(tmp_path, monkeypatch):
    candidate = _candidate("1", pine_ba=10, spruce_ba=5)
    stand = m.build_stand(candidate)
    config = m.ExtractionConfig(target_year=2018, altitude=150.0, ddy=1200.0)

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(m, "build_growth_and_yield_table", _boom)
    outcome = m.process_stand(stand, config, tmp_path)

    assert isinstance(outcome, m.StandSkipped)
    assert outcome.stand_id == stand.id
    assert "boom" in outcome.reason


def test_process_stand_leaves_no_stray_file_when_subdominant_fails(
    tmp_path, monkeypatch
):
    # A mixed stand where the dominant table builds fine but the subdominant
    # one fails: the outcome must be StandSkipped, and NO CSV -- not even the
    # already-computable dominant one -- may be left on disk, since that
    # would contradict the reported failure.
    candidate = _candidate("1", pine_ba=10, spruce_ba=5)
    stand = m.build_stand(candidate)
    config = m.ExtractionConfig(
        target_year=2018, altitude=150.0, ddy=1200.0, end_year=10
    )

    real_build = m.build_growth_and_yield_table
    call_count = {"n": 0}

    def _fail_on_second_call(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("boom on subdominant")
        return real_build(*args, **kwargs)

    monkeypatch.setattr(m, "build_growth_and_yield_table", _fail_on_second_call)
    outcome = m.process_stand(stand, config, tmp_path)

    assert isinstance(outcome, m.StandSkipped)
    assert list(tmp_path.glob("*.csv")) == []


def test_process_stand_leaves_no_stray_file_when_stand_data_validation_fails(
    tmp_path,
):
    # StandData is a pydantic model -- a degenerate stand.area (or any other
    # field failing its constraint) must raise before either CSV is written,
    # the same invariant test_process_stand_leaves_no_stray_file_when_
    # subdominant_fails already guards for the two growth tables.
    import dataclasses

    candidate = _candidate("1", pine_ba=10, spruce_ba=5)
    stand = dataclasses.replace(m.build_stand(candidate), area=0.0)
    config = m.ExtractionConfig(
        target_year=2018, altitude=150.0, ddy=1200.0, end_year=10
    )

    outcome = m.process_stand(stand, config, tmp_path)

    assert isinstance(outcome, m.StandSkipped)
    assert list(tmp_path.glob("*.csv")) == []


# %% build_stand's area


def test_build_stand_takes_area_from_the_area_column_not_the_geometry():
    # candidate.geometry is a 50 m radius disc, ~0.785 ha. The stand layer's
    # own area column is what the source reports, and that's what's kept
    # (CONTEXT.md, "Stand area") -- a deliberately different value here
    # proves the geometry isn't consulted.
    candidate = _candidate("1", pine_ba=10, area=1.234)
    stand = m.build_stand(candidate)
    assert stand.area == 1.234


# %% process_stand's StandData / dump_stand_data_document


def test_process_stand_returns_stand_data_with_the_shared_metadata_fields(tmp_path):
    candidate = _candidate("1", pine_ba=10, spruce_ba=5)
    stand = m.build_stand(candidate)
    config = m.ExtractionConfig(
        target_year=2018, altitude=150.0, ddy=1200.0, end_year=10
    )

    outcome = m.process_stand(stand, config, tmp_path)

    assert isinstance(outcome, m.StandWritten)
    stand_data = outcome.stand_data
    assert stand_data.site_fertility_class == stand.fertilityclass
    assert stand_data.x_ykj == stand.x_ykj
    assert stand_data.y_ykj == stand.y_ykj
    assert stand_data.stand_area == pytest.approx(stand.area)
    assert stand_data.polygon is not None
    assert stand_data.polygon.equals(stand.geometry)
    assert stand_data.main_group == m.MAINGROUP_FOREST_LAND
    assert stand_data.sub_group == stand.subgroup
    assert stand_data.developmentclass == stand.developmentclass
    assert stand_data.drainagestate == stand.drainagestate
    assert stand_data.soil_type == stand.soiltype
    assert stand_data.mean_age == pytest.approx(stand.stand_meanage)
    assert stand_data.basal_area == pytest.approx(stand.stand_basalarea)
    assert stand_data.mean_height == pytest.approx(stand.stand_meanheight)
    assert stand_data.mean_diameter == pytest.approx(stand.stand_meandiameter)


def test_process_stand_leaves_per_species_stand_data_fields_unset(tmp_path):
    """The per-species basal_area_*/stem_count_* fields are populated only by
    xml_to_allometry.py, whose consumer needs them -- the same single-source
    pattern drainagestate and developmentclass follow in the other direction.
    None here means "this source tool records nothing", not zero."""
    candidate = _candidate("1", pine_ba=10, spruce_ba=5)
    stand = m.build_stand(candidate)
    config = m.ExtractionConfig(
        target_year=2018, altitude=150.0, ddy=1200.0, end_year=10
    )

    outcome = m.process_stand(stand, config, tmp_path)

    assert isinstance(outcome, m.StandWritten)
    stand_data = outcome.stand_data
    assert stand_data.basal_area_pine is None
    assert stand_data.basal_area_spruce is None
    assert stand_data.basal_area_deciduous is None
    assert stand_data.stem_count_pine is None
    assert stand_data.stem_count_spruce is None
    assert stand_data.stem_count_deciduous is None


def test_process_stand_missing_soiltype_stays_none_in_stand_data(tmp_path):
    candidate = _candidate("1", pine_ba=10, soiltype=None)
    stand = m.build_stand(candidate)
    config = m.ExtractionConfig(
        target_year=2018, altitude=150.0, ddy=1200.0, end_year=10
    )

    outcome = m.process_stand(stand, config, tmp_path)

    assert isinstance(outcome, m.StandWritten)
    assert outcome.stand_data.soil_type is None


def test_process_stand_populates_allometry_file_per_layer_for_both_layers(tmp_path):
    candidate = _candidate("1", pine_ba=10, spruce_ba=5)
    stand = m.build_stand(candidate)
    config = m.ExtractionConfig(
        target_year=2018, altitude=150.0, ddy=1200.0, end_year=10
    )

    outcome = m.process_stand(stand, config, tmp_path)

    assert isinstance(outcome, m.StandWritten)
    allometry_file_per_layer = outcome.stand_data.allometry_file_per_layer
    dominant = allometry_file_per_layer[CanopyLayerName.dominant]
    assert dominant.file_path == outcome.dominant_csv
    assert dominant.species_id == stand.dominant_species
    subdominant = allometry_file_per_layer[CanopyLayerName.subdominant]
    assert subdominant.file_path == outcome.subdominant_csv
    assert subdominant.species_id == stand.subdominant_species


def test_process_stand_monoculture_allometry_file_per_layer_has_only_dominant(tmp_path):
    candidate = _candidate("1", pine_ba=10)
    stand = m.build_stand(candidate)
    config = m.ExtractionConfig(
        target_year=2018, altitude=150.0, ddy=1200.0, end_year=10
    )

    outcome = m.process_stand(stand, config, tmp_path)

    assert isinstance(outcome, m.StandWritten)
    allometry_file_per_layer = outcome.stand_data.allometry_file_per_layer
    assert CanopyLayerName.dominant in allometry_file_per_layer
    assert CanopyLayerName.subdominant not in allometry_file_per_layer


def test_dump_stand_data_document_is_the_shared_function():
    # dump_stand_data_document now lives in tools.shared_allometry_tool_utils.
    # stand_data, imported here instead of a local copy -- its write-to-disk
    # behavior is tested once, in test_shared_allometry_tool_utils.py.
    assert m.dump_stand_data_document is shared_dump_stand_data_document


# %% End-to-end pipeline, against a tiny synthetic .gpkg


def _write_synthetic_gpkg(
    path: Path, crs: str | None = shared_utils.SOURCE_CRS
) -> None:
    """A minimal 3-layer gpkg exercising load_gpkg_layers + the whole
    filter/merge/aggregate chain: two stands that survive filtering (one
    pine-only monoculture, one pine+spruce mix) and three that each fail a
    different filter/step, so the pipeline's skip-reporting is exercised too."""
    helsinki_area = Point(385000, 6685000).buffer(50)

    stand = gpd.GeoDataFrame(
        [
            _stand_row(1, subgroup=2, drainagestate=7, fertilityclass=3),  # survives
            _stand_row(2, subgroup=2, drainagestate=8, fertilityclass=4),  # survives
            _stand_row(3, maingroup=2),  # excluded: not forest land
            _stand_row(4, subgroup=2, drainagestate=6),  # excluded: undrained
            _stand_row(
                5, subgroup=2, drainagestate=7, fertilityclass=3
            ),  # excluded: no type=1 snapshot in target year
        ],
        geometry="geometry",
        # The geometry below is EPSG:3067. Left unlabelled only when the
        # test asks for a layer with no CRS at all.
        crs=shared_utils.SOURCE_CRS if crs is not None else None,
    )
    stand["geometry"] = [helsinki_area] * 5
    if crs is not None:
        # The same stands, really in `crs`: reprojected, not just relabelled.
        stand = stand.to_crs(crs)

    treestand = pd.DataFrame(
        [
            _treestand_row(1, 101, "2018-06-01", 1),
            _treestand_row(2, 102, "2018-07-01", 1),
            _treestand_row(3, 103, "2018-06-01", 1),
            _treestand_row(4, 104, "2018-06-01", 1),
            _treestand_row(5, 105, "2015-06-01", 1),  # wrong year -- no 2018 match
        ]
    )

    treestratum = pd.DataFrame(
        [
            # Stand 1 (treestandid 101): pine only -- a monoculture.
            {
                "treestandid": 101,
                "treespecies": 1,
                "age": 40.0,
                "basalarea": 20.0,
                "stemcount": 500.0,
                "meandiameter": 18.0,
                "meanheight": 16.0,
            },
            # Stand 2 (treestandid 102): pine + spruce mix.
            {
                "treestandid": 102,
                "treespecies": 1,
                "age": 35.0,
                "basalarea": 12.0,
                "stemcount": 400.0,
                "meandiameter": 16.0,
                "meanheight": 14.0,
            },
            {
                "treestandid": 102,
                "treespecies": 2,
                "age": 30.0,
                "basalarea": 8.0,
                "stemcount": 300.0,
                "meandiameter": 14.0,
                "meanheight": 12.0,
            },
        ]
    )

    stand.to_file(path, layer="stand", driver="GPKG")
    # treestand/treestratum carry no geometry -- write as plain (non-spatial)
    # gpkg layers, same as gpd.read_file returns them for metsakeskus.py's
    # real MV_Uusimaa.gpkg (attribute-only tables read back as GeoDataFrames
    # with an all-None geometry column, which this tool never touches for
    # those two layers).
    gpd.GeoDataFrame(treestand).to_file(path, layer="treestand", driver="GPKG")
    gpd.GeoDataFrame(treestratum).to_file(path, layer="treestratum", driver="GPKG")


def test_full_pipeline_end_to_end_with_synthetic_gpkg(tmp_path):
    gpkg_path = tmp_path / "synthetic.gpkg"
    _write_synthetic_gpkg(gpkg_path)
    config = m.ExtractionConfig(
        target_year=2018, altitude=150.0, ddy=1200.0, end_year=10
    )

    layers = m.load_gpkg_layers(gpkg_path)
    assert len(layers.stand) == 5

    filtered_stand = m.filter_stands_by_site_attributes(
        layers.stand, config.fertilityclass_filter
    )
    assert sorted(filtered_stand["standid"]) == [
        1,
        2,
        5,
    ]  # 3 and 4 excluded at this step

    stand_ids = set(pd.to_numeric(filtered_stand["standid"]).astype(int))
    snapshot = m.select_target_year_snapshot(
        layers.treestand, stand_ids, config.target_year
    )
    assert sorted(snapshot["standid"]) == [1, 2]  # 5 excluded: no exact 2018 match

    merged = m.attach_stand_attributes(snapshot, filtered_stand)
    merged_filtered = m.filter_by_developmentclass(
        merged, config.developmentclass_filter
    )
    assert sorted(merged_filtered["standid"]) == [1, 2]

    candidates, structural_skips = m.build_stand_candidates(
        merged_filtered, layers.treestratum
    )
    assert len(structural_skips) == 0
    assert len(candidates) == 2

    viable, ba_skips = m.partition_viable_candidates(candidates)
    assert len(ba_skips) == 0
    assert len(viable) == 2

    parsed_stands = [m.build_stand(c) for c in viable]
    stands_by_id = {str(s.id): s for s in parsed_stands}

    # Stand 1: pine-only monoculture. Per docs/adr/0002, the subdominant is
    # the genuine zero-BA runner-up (spruce or deciduous), never a duplicate
    # of the dominant (pine).
    stand_1 = stands_by_id["1"]
    assert stand_1.dominant_species == 1
    assert stand_1.subdominant_species != 1
    assert stand_1.stand_basalarea == pytest.approx(20.0)

    # Stand 2: pine (12 m2/ha) + spruce (8 m2/ha) -- pine dominant, spruce subdominant.
    stand_2 = stands_by_id["2"]
    assert stand_2.dominant_species == 1
    assert stand_2.subdominant_species == 2
    assert stand_2.stand_basalarea == pytest.approx(20.0)

    # tmp_path plays the project root here: the CSVs go where a real run
    # puts them, inside the inputs/ folder stand_data.json is written to --
    # the document refuses to record a file outside it (docs/adr/0005).
    allometry_dir = tmp_path / "inputs" / "allometry"
    allometry_dir.mkdir(parents=True)
    outcomes = {
        str(o.stand_id): o
        for o in (m.process_stand(s, config, allometry_dir) for s in parsed_stands)
    }
    outcome_1, outcome_2 = outcomes["1"], outcomes["2"]
    assert isinstance(outcome_1, m.StandWritten)
    assert isinstance(outcome_2, m.StandWritten)

    # Stand 1 is a monoculture: dominant only, no subdominant file.
    assert outcome_1.dominant_csv.exists()
    assert outcome_1.subdominant_csv is None

    # Stand 2 is a real mix: both layers get a file.
    assert outcome_2.dominant_csv.exists()
    assert outcome_2.subdominant_csv is not None
    assert outcome_2.subdominant_csv.exists()

    json_path = tmp_path / "inputs" / "stand_data.json"
    m.dump_stand_data_document(
        output_path=json_path,
        document=StandDataDocument(
            crs=shared_utils.SOURCE_CRS,
            altitude=config.altitude,
            ddy=config.ddy,
            # Built from the two locals the isinstance asserts above already
            # narrowed to StandWritten, not from outcomes.values(): only a
            # StandWritten carries stand_data, and main() likewise builds
            # this document from its written list alone.
            stands={
                outcome.stand_id: outcome.stand_data
                for outcome in (outcome_1, outcome_2)
            },
        ),
    )
    assert json_path.exists()
    reloaded = load_stand_data_document_from_json(json_path)
    assert set(str(sid) for sid in reloaded.stands) == {"1", "2"}
    # Allometry paths come back absolute, pointing at the files written above.
    assert (
        reloaded.stands[StandID("2")]
        .allometry_file_per_layer[CanopyLayerName.subdominant]
        .file_path
        == outcome_2.subdominant_csv
    )
    # The source's own area column (_stand_row's 1.5 ha), and the real
    # polygon, both survive the JSON round-trip.
    assert reloaded.stands[StandID("1")].stand_area == 1.5
    reloaded_polygon = reloaded.stands[StandID("1")].polygon
    assert reloaded_polygon is not None
    assert reloaded_polygon.equals(layers.stand.geometry.iloc[0])


# %% stand_layer_in_source_crs


def test_stand_layer_in_source_crs_returns_a_source_crs_layer_as_is():
    stand = _make_stand_gdf([_stand_row(1)])
    assert m.stand_layer_in_source_crs(stand) is stand


def test_stand_layer_in_source_crs_reprojects_another_crs():
    in_source_crs = _make_stand_gdf([_stand_row(1)])
    in_wgs84 = in_source_crs.to_crs("EPSG:4326")

    result = m.stand_layer_in_source_crs(in_wgs84)

    assert result.crs == shared_utils.SOURCE_CRS
    assert result.geometry.iloc[0].equals_exact(
        in_source_crs.geometry.iloc[0], tolerance=1e-3
    )


def test_stand_layer_in_source_crs_fails_without_a_crs():
    stand = gpd.GeoDataFrame([_stand_row(1)], geometry="geometry", crs=None)
    with pytest.raises(ValueError, match="no CRS"):
        m.stand_layer_in_source_crs(stand)


# %% main


def _run_main(monkeypatch, gpkg_path, project_dir):
    (project_dir / "inputs" / "config.toml").write_text(
        "target_year = 2018\naltitude = 150.0\nddy = 1200.0\nend_year = 10\n"
    )
    monkeypatch.setattr(
        "sys.argv",
        ["metsakeskus_to_allometry.py", str(gpkg_path), f"--project-dir={project_dir}"],
    )
    m.main()


def test_main_writes_crs_and_polygons_into_stand_data_json(
    monkeypatch, tmp_path, project_dir
):
    gpkg_path = tmp_path / "synthetic.gpkg"
    _write_synthetic_gpkg(gpkg_path)

    _run_main(monkeypatch, gpkg_path, project_dir)

    json_path = project_dir / "inputs" / "stand_data.json"
    assert '"polygon":"POLYGON ((' in json_path.read_text()
    document = load_stand_data_document_from_json(json_path)
    assert document.crs == shared_utils.SOURCE_CRS
    assert all(stand.polygon is not None for stand in document.stands.values())


def test_main_records_allometry_paths_relative_to_stand_data_json(
    monkeypatch, tmp_path, project_dir
):
    # Run from project_dir's parent with a relative --project-dir: the
    # recorded paths must not depend on either (docs/adr/0005).
    gpkg_path = tmp_path / "synthetic.gpkg"
    _write_synthetic_gpkg(gpkg_path)
    monkeypatch.chdir(project_dir.parent)

    _run_main(monkeypatch, gpkg_path, Path(project_dir.name))

    json_path = project_dir / "inputs" / "stand_data.json"
    raw = json.loads(json_path.read_text())
    recorded = {
        layer["file_path"]
        for stand in raw["stands"].values()
        for layer in stand["allometry_file_per_layer"].values()
    }
    assert recorded
    assert all(path.startswith("allometry/") for path in recorded)
    document = load_stand_data_document_from_json(json_path)
    for stand in document.stands.values():
        for entry in stand.allometry_file_per_layer.values():
            assert entry.file_path.parent == project_dir / "inputs" / "allometry"
            assert entry.file_path.exists()


def test_main_reprojects_a_stand_layer_in_another_crs(
    monkeypatch, tmp_path, project_dir
):
    reference_gpkg = tmp_path / "reference.gpkg"
    _write_synthetic_gpkg(reference_gpkg)
    wgs84_gpkg = tmp_path / "wgs84.gpkg"
    _write_synthetic_gpkg(wgs84_gpkg, crs="EPSG:4326")

    _run_main(monkeypatch, wgs84_gpkg, project_dir)

    document = load_stand_data_document_from_json(
        project_dir / "inputs" / "stand_data.json"
    )
    assert document.crs == shared_utils.SOURCE_CRS
    reference_polygon = gpd.read_file(reference_gpkg, layer="stand").geometry.iloc[0]
    for stand in document.stands.values():
        assert stand.polygon is not None
        assert stand.polygon.equals_exact(reference_polygon, tolerance=1e-3)


def test_main_fails_the_run_on_a_stand_layer_without_a_crs(
    monkeypatch, tmp_path, project_dir
):
    gpkg_path = tmp_path / "synthetic.gpkg"
    # geopandas warns when writing a layer with no CRS -- that's the point.
    with pytest.warns(UserWarning, match="crs"):
        _write_synthetic_gpkg(gpkg_path, crs=None)

    with pytest.raises(ValueError, match="no CRS"):
        _run_main(monkeypatch, gpkg_path, project_dir)

    assert not (project_dir / "inputs" / "stand_data.json").exists()
    assert not (project_dir / "inputs" / "allometry").exists()


# %% plan_stand_outputs / csv_counts


def test_plan_stand_outputs_names_both_layers(tmp_path):
    stand = m.build_stand(_candidate("1", pine_ba=10, spruce_ba=5))

    plan = m.plan_stand_outputs(stand, tmp_path)

    assert plan.stand_id == stand.id
    assert plan.dominant_csv == tmp_path / "1_dominant.csv"
    assert plan.subdominant_csv == tmp_path / "1_subdominant.csv"


def test_plan_stand_outputs_monoculture_has_no_subdominant(tmp_path):
    # The same rule process_stand applies: a second-ranked species with zero
    # basal area gets no file at all (docs/adr/0002).
    stand = m.build_stand(_candidate("1", pine_ba=10))

    plan = m.plan_stand_outputs(stand, tmp_path)

    assert plan.dominant_csv == tmp_path / "1_dominant.csv"
    assert plan.subdominant_csv is None


def test_plan_stand_outputs_touches_nothing_on_disk(tmp_path):
    # The premise of --dry-run: planning is pure, so it can name a folder that
    # does not exist without bringing it into being.
    stand = m.build_stand(_candidate("1", pine_ba=10, spruce_ba=5))

    m.plan_stand_outputs(stand, tmp_path / "never_created")

    assert not (tmp_path / "never_created").exists()
    assert list(tmp_path.iterdir()) == []


def test_plan_stand_outputs_agrees_with_what_process_stand_writes(tmp_path):
    # The point of routing both paths through plan_stand_outputs: what a dry
    # run reports is what a real run then puts on disk, name for name.
    stand = m.build_stand(_candidate("1", pine_ba=10, spruce_ba=5))
    config = m.ExtractionConfig(
        target_year=2018, altitude=150.0, ddy=1200.0, end_year=10
    )

    plan = m.plan_stand_outputs(stand, tmp_path)
    outcome = m.process_stand(stand, config, tmp_path)

    assert isinstance(outcome, m.StandWritten)
    assert outcome.dominant_csv == plan.dominant_csv
    assert outcome.subdominant_csv == plan.subdominant_csv
    assert plan.subdominant_csv is not None
    assert sorted(path.name for path in tmp_path.glob("*.csv")) == sorted(
        [plan.dominant_csv.name, plan.subdominant_csv.name]
    )


def test_csv_counts_counts_dominant_and_subdominant_separately(tmp_path):
    plans = [
        m.plan_stand_outputs(
            m.build_stand(_candidate("1", pine_ba=10, spruce_ba=5)), tmp_path
        ),
        m.plan_stand_outputs(m.build_stand(_candidate("2", pine_ba=10)), tmp_path),
    ]

    # Two stands -> two dominant files, but only the mixed stand adds a
    # subdominant one.
    assert m.csv_counts(plans) == (2, 1)


# %% CLI


@pytest.fixture
def dummy_gpkg_file(tmp_path):
    gpkg_path = tmp_path / "stands.gpkg"
    gpkg_path.write_bytes(b"")
    return gpkg_path


@pytest.fixture
def dummy_config_file(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text("target_year = 2018\naltitude = 150.0\nddy = 1200.0\n")
    return config_path


@pytest.fixture
def project_dir(tmp_path):
    # --project-dir is the project root, and must already exist (shared
    # valid_existing_directory). Everything this tool reads and writes lives
    # in its inputs/ folder: config.toml going in, the allometry CSVs and
    # stand_data.json coming out.
    path = tmp_path / "myproject"
    (path / "inputs").mkdir(parents=True)
    return path


def _run_parse_CLI_arguments(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["metsakeskus_to_allometry.py", *argv])
    return m.parse_CLI_arguments()


# m.valid_existing_directory (--project-dir's validator) is now the shared
# tools.shared_allometry_tool_utils.input_validation function -- its own
# behavior (empty/missing/not-a-directory) is covered directly in
# test_shared_allometry_tool_utils.py. This just checks metsakeskus_to_
# allometry.py actually wires it up as --project-dir's type.
def test_valid_existing_directory_accepts_an_existing_directory(project_dir):
    assert m.valid_existing_directory(str(project_dir)) == project_dir


def test_parse_CLI_arguments_requires_project_dir(
    monkeypatch, dummy_gpkg_file, dummy_config_file, capsys
):
    with pytest.raises(SystemExit):
        _run_parse_CLI_arguments(
            monkeypatch, [str(dummy_gpkg_file), f"--config={dummy_config_file}"]
        )
    assert "--project-dir" in capsys.readouterr().err


def test_parse_CLI_arguments_reports_a_missing_default_config(
    monkeypatch, dummy_gpkg_file, project_dir, capsys
):
    # --config is optional now: omitting it means "look for config.toml
    # inside --project-dir", and there is none here -- report that, pointing
    # back at --config as the way out, rather than a bare file-not-found.
    with pytest.raises(SystemExit):
        _run_parse_CLI_arguments(
            monkeypatch, [str(dummy_gpkg_file), f"--project-dir={project_dir}"]
        )
    stderr = capsys.readouterr().err
    assert str(project_dir / "inputs" / "config.toml") in stderr
    assert "--config" in stderr


def test_parse_CLI_arguments_finds_config_toml_inside_project_dir_by_default(
    monkeypatch, dummy_gpkg_file, project_dir
):
    (project_dir / "inputs" / "config.toml").write_text(
        "target_year = 2018\naltitude = 150.0\nddy = 1200.0\n"
    )
    cli_args = _run_parse_CLI_arguments(
        monkeypatch, [str(dummy_gpkg_file), f"--project-dir={project_dir}"]
    )
    assert cli_args.config_path == project_dir / "inputs" / "config.toml"
    assert cli_args.config.target_year == 2018


def test_parse_CLI_arguments_explicit_config_overrides_the_default_lookup(
    monkeypatch, dummy_gpkg_file, dummy_config_file, project_dir
):
    # A config.toml also happens to sit inside project_dir; an explicit
    # --config must still win over the default lookup.
    (project_dir / "inputs" / "config.toml").write_text(
        "target_year = 1999\naltitude = 10.0\nddy = 600.0\n"
    )
    cli_args = _run_parse_CLI_arguments(
        monkeypatch,
        [
            str(dummy_gpkg_file),
            f"--config={dummy_config_file}",
            f"--project-dir={project_dir}",
        ],
    )
    assert cli_args.config_path == dummy_config_file
    assert (
        cli_args.config.target_year == 2018
    )  # from dummy_config_file, not project_dir's own


def test_parse_CLI_arguments_rejects_a_second_positional_argument(
    monkeypatch, dummy_gpkg_file, dummy_config_file, project_dir, tmp_path, capsys
):
    # The output folder used to be an optional second positional. It is gone:
    # <project-dir>/inputs/allometry/ is now the only place output can land,
    # so a leftover invocation must fail loudly rather than be ignored.
    with pytest.raises(SystemExit):
        _run_parse_CLI_arguments(
            monkeypatch,
            [
                str(dummy_gpkg_file),
                str(tmp_path / "custom_out"),
                f"--config={dummy_config_file}",
                f"--project-dir={project_dir}",
            ],
        )
    assert "unrecognized arguments" in capsys.readouterr().err


def test_parse_CLI_arguments_creates_no_output_folder(
    monkeypatch, dummy_gpkg_file, dummy_config_file, project_dir
):
    # Creating the folder is main()'s job now, so a run that dies while
    # reading or filtering leaves nothing behind to block the next attempt.
    _run_parse_CLI_arguments(
        monkeypatch,
        [
            str(dummy_gpkg_file),
            f"--config={dummy_config_file}",
            f"--project-dir={project_dir}",
        ],
    )
    assert not (project_dir / "inputs" / "allometry").exists()


def test_parse_CLI_arguments_dry_run_defaults_to_false(
    monkeypatch, dummy_gpkg_file, dummy_config_file, project_dir
):
    cli_args = _run_parse_CLI_arguments(
        monkeypatch,
        [
            str(dummy_gpkg_file),
            f"--config={dummy_config_file}",
            f"--project-dir={project_dir}",
        ],
    )
    assert cli_args.dry_run is False


def test_parse_CLI_arguments_dry_run_creates_no_output_folder(
    monkeypatch, dummy_gpkg_file, dummy_config_file, project_dir
):
    # The flag's core promise: a dry run puts nothing on disk, not even the
    # empty folder that would then block the real run behind a new
    # --project-dir.
    cli_args = _run_parse_CLI_arguments(
        monkeypatch,
        [
            str(dummy_gpkg_file),
            f"--config={dummy_config_file}",
            f"--project-dir={project_dir}",
            "--dry-run",
        ],
    )
    assert cli_args.dry_run is True
    assert not (project_dir / "inputs" / "allometry").exists()


def test_parse_CLI_arguments_refuses_existing_default_output_dir(
    monkeypatch, dummy_gpkg_file, dummy_config_file, project_dir, capsys
):
    # A folder already there -- e.g. left over from a previous run -- must
    # stop the tool instead of being silently written into (see the removed
    # clear_previous_outputs: it used to delete whatever was already there).
    (project_dir / "inputs" / "allometry").mkdir()

    with pytest.raises(SystemExit):
        _run_parse_CLI_arguments(
            monkeypatch,
            [
                str(dummy_gpkg_file),
                f"--config={dummy_config_file}",
                f"--project-dir={project_dir}",
            ],
        )
    stderr = capsys.readouterr().err
    assert "already exists" in stderr
    assert "--project-dir" in stderr


def test_parse_CLI_arguments_dry_run_still_refuses_existing_output_dir(
    monkeypatch, dummy_gpkg_file, dummy_config_file, project_dir, capsys
):
    # A dry run reports what the real run would do -- and what the real run
    # would do here is refuse to start. Catching that is exactly the point.
    (project_dir / "inputs" / "allometry").mkdir()

    with pytest.raises(SystemExit):
        _run_parse_CLI_arguments(
            monkeypatch,
            [
                str(dummy_gpkg_file),
                f"--config={dummy_config_file}",
                f"--project-dir={project_dir}",
                "--dry-run",
            ],
        )
    assert "already exists" in capsys.readouterr().err


def test_parse_CLI_arguments_blocks_out_of_range_altitude(
    monkeypatch, dummy_gpkg_file, project_dir, tmp_path, capsys
):
    config_path = tmp_path / "config.toml"
    config_path.write_text("target_year = 2018\naltitude = 1500.0\nddy = 1200.0\n")
    with pytest.raises(SystemExit):
        _run_parse_CLI_arguments(
            monkeypatch,
            [
                str(dummy_gpkg_file),
                f"--config={config_path}",
                f"--project-dir={project_dir}",
            ],
        )
    stderr = capsys.readouterr().err
    assert "altitude" in stderr
    assert "--allow-out-of-range-values" in stderr


def test_parse_CLI_arguments_allows_out_of_range_with_override(
    monkeypatch, dummy_gpkg_file, project_dir, tmp_path, capsys
):
    config_path = tmp_path / "config.toml"
    config_path.write_text("target_year = 2018\naltitude = 1500.0\nddy = 1200.0\n")
    cli_args = _run_parse_CLI_arguments(
        monkeypatch,
        [
            str(dummy_gpkg_file),
            f"--config={config_path}",
            f"--project-dir={project_dir}",
            "--allow-out-of-range-values",
        ],
    )
    assert cli_args.config.altitude == 1500.0
    assert "Warning" in capsys.readouterr().out


@given(
    coords=st.lists(
        st.tuples(
            st.floats(
                min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
            ),
            st.floats(
                min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
            ),
        ),
        min_size=3,
    )
)
def test_estimate_stemcount_never_negative(coords):
    # Not really about coords -- reuses hypothesis' float generator to fuzz
    # estimate_stemcount's numeric edges cheaply.
    for x, _y in coords:
        assert m.estimate_stemcount(abs(x), abs(x) + 1) >= 0

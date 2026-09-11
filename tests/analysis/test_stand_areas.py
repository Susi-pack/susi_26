from pathlib import Path

import pytest

from analysis.optimization.stand_areas import (
    areas_from_stand_datas,
    stand_areas_for_project,
)
from susi.io.load_output_data import StandID

# Shaped like the "stand_datas" entry of Paroninkorpi's extra_xml_info.json:
# stand number as a string key, and many more properties per stand than the
# area this module cares about.
_STAND_DATAS = {
    "1": {"area": 2.4, "mean_height": 22.1},
    "2": {"area": 1.9, "mean_height": 18.7},
    "10": {"area": 0.5, "mean_height": 20.0},
}


class TestAreasFromStandDatas:
    def test_picks_the_area_of_each_requested_stand(self):
        areas_ha = areas_from_stand_datas(
            stand_datas=_STAND_DATAS,
            stand_ids=[StandID("stand_1"), StandID("stand_10")],
        )

        assert areas_ha == {StandID("stand_1"): 2.4, StandID("stand_10"): 0.5}

    def test_result_is_keyed_by_stand_id_not_by_stand_number(self):
        areas_ha = areas_from_stand_datas(
            stand_datas=_STAND_DATAS, stand_ids=[StandID("stand_2")]
        )

        assert list(areas_ha) == [StandID("stand_2")]

    def test_stand_missing_from_the_file_raises_naming_that_stand(self):
        # A bare KeyError here would otherwise surface much later, inside
        # core.build_optimization_array, with no clue which stand was missing.
        with pytest.raises(ValueError, match="stand_7"):
            areas_from_stand_datas(
                stand_datas=_STAND_DATAS, stand_ids=[StandID("stand_7")]
            )

    def test_stand_folder_not_named_stand_number_raises(self):
        with pytest.raises(ValueError, match="not_a_stand"):
            areas_from_stand_datas(
                stand_datas=_STAND_DATAS, stand_ids=[StandID("not_a_stand")]
            )


class TestStandAreasForProject:
    def test_project_other_than_paroninkorpi_raises_and_points_at_the_issue(self):
        # Sourcing areas for other projects is #216; until then this must
        # fail loudly rather than guess.
        with pytest.raises(ValueError, match="#216"):
            stand_areas_for_project(project_dirpath=Path("outputs/some_other_site"))

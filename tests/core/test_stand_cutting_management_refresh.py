# Tests for Stand.apply_cutting_management refreshing stand-level aggregates
# (issue #180). Scope: after a real (non-mocked) do_thinning/do_clearcut,
# every stand-level field that Stand.update() sums from the canopy layers
# (hdom, leafarea, basalarea, stems, volume, biomass, mean_diameter, N/P/K
# demand, ...) must already reflect the post-cut canopy-layer state. Before
# the fix, Canopylayer.do_thinning/do_clearcut correctly updated the canopy
# layer's own state, but nothing re-summed that into the stand-level fields
# -- they stayed at their pre-cut values for a full simulated year, until
# the *next* year's regular stand.assimilate()/update() cycle caught up.
#
# Compare with test_stand_cutting_management_dispatch.py, which only checks
# routing (right Canopylayer method called with the right args) via
# monkeypatched fakes, and never exercises real per-layer state changes or
# Stand.update()/update_logging().
from pathlib import Path

import numpy as np

from susi.core.stand import Stand
from susi.io.susi_parameter_model import (
    AllometryFileAndSpecies,
    CanopyLayerAllometry,
    CanopyLayerName,
    ClearCut,
    CuttingManagementParams,
    LocationsForPhotoParams,
    Thinning,
    get_photo_parameters_by_location,
)

DATA_DIR = Path(__file__).parent.parent / "data"
N = 5
SFC = np.ones(N, dtype=int) * 4


def _make_stand(age: float = 70.0) -> Stand:
    allometry_params = CanopyLayerAllometry(
        allometry_file_registry={
            1: AllometryFileAndSpecies(
                file_path=DATA_DIR / "test_allometry.csv", species_id=1
            )
        },
        pointers={
            CanopyLayerName.dominant: [1] * N,
            CanopyLayerName.subdominant: None,
            CanopyLayerName.under: None,
        },
    )
    agearr = {
        "dominant": np.full(N, age),
        "subdominant": np.full(N, age),
        "under": np.full(N, age),
    }
    stand = Stand(
        n_scenarios=1,
        n_yrs=4,
        n_cols=N,
        sfc=SFC,
        agearr=agearr,
        allometry_params=allometry_params,
        photopara=get_photo_parameters_by_location(
            location=LocationsForPhotoParams("All_data")
        ),
    )
    stand.update()
    return stand


def _regeneration_allometry() -> CanopyLayerAllometry:
    """age must start at 1 -- see ClearCut.new_allometry_includes_age_one."""
    return CanopyLayerAllometry(
        allometry_file_registry={
            1: AllometryFileAndSpecies(
                file_path=DATA_DIR / "post_clearcut_allom.csv", species_id=1
            )
        },
        pointers={
            CanopyLayerName.dominant: [1] * N,
            CanopyLayerName.subdominant: None,
            CanopyLayerName.under: None,
        },
    )


def _full_clearcut(yr: int) -> CuttingManagementParams:
    return CuttingManagementParams(
        application_yr=yr,
        management_type=ClearCut(
            new_growth_allometry=_regeneration_allometry(),
            strips_to_cut=[True] * N,
        ),
    )


STAND_AGGREGATE_FIELDS = (
    "hdom",
    "leafarea",
    "basalarea",
    "stems",
    "volume",
    "biomass",
    "mean_diameter",
    "n_demand",
    "p_demand",
    "k_demand",
)


class TestClearCutRefreshesStandAggregates:
    def test_hdom_drops_to_the_new_saplings_height(self):
        stand = _make_stand(age=70.0)
        pre_cut_hdom = stand.hdom.copy()

        stand.apply_cutting_management(_full_clearcut(2005), yr=2005, sfc=SFC)

        assert not np.allclose(stand.hdom, pre_cut_hdom)
        # dominant is the only real layer here (subdominant/under pointers
        # are None), so the stand-level hdom must match it exactly once the
        # aggregate is refreshed.
        np.testing.assert_allclose(stand.hdom, stand.dominant.hdom)

    def test_stand_level_fields_already_match_a_fresh_update(self):
        """Idempotency check: calling stand.update() again right after the
        cut must not change anything -- if it does, apply_cutting_management
        left the aggregates stale."""
        stand = _make_stand(age=70.0)
        stand.apply_cutting_management(_full_clearcut(2005), yr=2005, sfc=SFC)

        before = {
            field: getattr(stand, field).copy() for field in STAND_AGGREGATE_FIELDS
        }
        stand.update()
        for field, value in before.items():
            np.testing.assert_allclose(getattr(stand, field), value, err_msg=field)

    def test_harvested_volume_from_the_felled_stand_is_not_lost(self):
        """Guards the ordering hazard: Stand.update()'s reset_vars() zeroes
        the harvested_*/*_lresid accumulator fields, so if
        apply_cutting_management's internal refresh ran *after*
        update_logging() instead of before it, the harvest just computed for
        this cut would get wiped back to zero."""
        stand = _make_stand(age=70.0)
        stand.apply_cutting_management(_full_clearcut(2005), yr=2005, sfc=SFC)
        stand.update_logging()

        assert np.all(stand.harvested_volume > 0)


class TestThinningRefreshesStandAggregates:
    def test_stems_drop_after_thinning(self):
        stand = _make_stand(age=70.0)
        pre_cut_stems = stand.stems.copy()

        stand.apply_cutting_management(
            CuttingManagementParams(
                application_yr=2005,
                management_type=Thinning(target_basal_area={CanopyLayerName.dominant: 12}),
            ),
            yr=2005,
            sfc=SFC,
        )

        assert np.all(stand.stems < pre_cut_stems)
        np.testing.assert_allclose(stand.stems, stand.dominant.stems)

# Tests for Canopylayer.do_thinning across multiple allometry zones (issue
# #195). Scope: do_thinning must not corrupt (or crash on) a layer with more
# than one allometry zone.
#
# History: #195 diagnosed two bugs in do_thinning's per-zone loop -- (a) the
# self.remaining_share assignment overwrote the *entire* per-column array
# using only the current zone's own bm_to_ba curve (last zone processed wins
# for every column), and (b) self.update(self.biomass) was re-run once per
# zone instead of once after the loop. Commit 6c68301 ("Fixes #195") fixed
# (b) (dedented self.update() below the loop) but only half-fixed (a): it
# scoped the *assignment target* to zone.cols, but left the right-hand side
# reading the whole-layer self.biomass/self.stems instead of the zone's own
# cols-sliced subset. For a single-zone layer (zone.cols covers every
# column) the shapes happen to match, so nothing looked wrong -- every
# config in this repo before now was single-zone (matching #195's own
# "currently unreachable" note) -- but a genuine multi-zone layer hits a
# shape mismatch outright.
from pathlib import Path

import numpy as np

from susi.core.stand import Stand
from susi.io.susi_parameter_model import (
    AllometryFileAndSpecies,
    CanopyLayerAllometry,
    CanopyLayerName,
    LocationsForPhotoParams,
    get_photo_parameters_by_location,
)

DATA_DIR = Path(__file__).parent.parent / "data"
N = 5
# zone 1: cols 0,1 (sfc=1); zone 2: cols 2,3,4 (sfc=4) -- two real zones, so a
# curve/array mixed across zones is visibly wrong, not a numerical coincidence.
POINTERS = [1, 1, 2, 2, 2]
SFC = np.array([1, 1, 4, 4, 4])


def _make_multizone_stand(age: float = 70.0) -> Stand:
    allometry_params = CanopyLayerAllometry(
        allometry_file_registry={
            1: AllometryFileAndSpecies(
                file_path=DATA_DIR / "test_allometry.csv", species_id=1
            ),
            2: AllometryFileAndSpecies(
                file_path=DATA_DIR / "test_allometry.csv", species_id=1
            ),
        },
        pointers={
            CanopyLayerName.dominant: POINTERS,
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
        growth_mode="dynamic",
    )
    stand.update()
    return stand


class TestDoThinningMultiZone:
    def test_does_not_crash_with_multiple_zones(self):
        """Regression test for #195's incompletely-fixed shape mismatch:
        ValueError: shape mismatch: value array of shape (5,) could not be
        broadcast to indexing result of shape (2,)."""
        stand = _make_multizone_stand()
        stand.dominant.do_thinning(yr=2005, nut_stat=stand.nut_stat, to_ba=12.0)

    def test_remaining_share_uses_each_zones_own_biomass_and_stems(self):
        """Each zone's remaining_share must come from that zone's own
        allometry curve applied to that zone's own (pre-thinning) biomass
        and stems -- not the whole-layer arrays. Computes the expected value
        independently per zone and compares, so it doesn't rely on the two
        zones' curves happening to agree."""
        stand = _make_multizone_stand()
        layer = stand.dominant
        biomass_before = layer.biomass.copy()
        stems_before = layer.stems.copy()
        to_ba = 12.0

        layer.do_thinning(yr=2005, nut_stat=stand.nut_stat, to_ba=to_ba)

        for zone in layer.zones:
            cols = zone.cols
            expected = to_ba / (
                zone.allometry.functions.bm_to_ba(biomass_before[cols])
                * stems_before[cols]
            )
            np.testing.assert_allclose(layer.remaining_share[cols], expected)

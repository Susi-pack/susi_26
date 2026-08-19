# Model-level tests for the CuttingManagementParams API (issue #171).
#
# Scope: this file only exercises the Pydantic model (SiteParams.cutting,
# ClearCutParams, ThinningParams, and the old-API-rejection/strips-length
# validators). It intentionally does NOT test the cutting *behavior*
# (Canopylayer.cutting() dispatch, strip-selective harvesting, post-cut
# allometry switch-over) — that lives in susi's OOP core and hasn't been
# ported yet. Compare with supersusi's tests/tests_cutting_management.py,
# whose TestClearCutSelection class tests exactly that runtime behavior
# against supersusi's functional cut_layer()/apply_allometry(); there is no
# susi-side equivalent of those functions yet, so that class has no
# counterpart here.
import datetime
from pathlib import Path

import pytest

from susi.io.susi_parameter_model import (
    AllometryParams,
    CanopyLayerAllometryPointers,
    CanopyParams,
    ClearCutParams,
    OrganicLayerParams,
    OutputParams,
    PeatTemperatureParams,
    PeatTypes,
    SimulationConfig,
    SiteParams,
    SusiParams,
    ThinningParams,
    TreeSpecies,
    WeatherParams,
    get_photo_parameters_by_location,
    LocationsForPhotoParams,
    h_mor_from_drainage_and_mass_mor_Pitkanen,
)

DATA_DIR = Path(__file__).parent / "data"


def _make_susi_params(*, cutting, n: int = 5):
    """Build a minimal-but-valid SusiParams with the given cutting intervention.

    Mirrors the `valid_susi_params` fixture in test_susi_model.py, parameterized
    on `cutting` so each test only has to specify the piece it cares about.
    """
    return SusiParams(
        weather_parameters=WeatherParams(
            FMI_weather_filepath=DATA_DIR / "weather.csv",
        ),
        simulation_config=SimulationConfig(
            start_date=datetime.datetime(2004, 1, 1),
            end_date=datetime.datetime(2007, 12, 31),
        ),
        allometry_parameters=AllometryParams(
            allometry_dir_path=DATA_DIR,
            dominant={1: "test_allometry.xlsx"},
            subdominant={0: "test_allometry.xlsx"},
            under={0: "test_allometry.xlsx"},
        ),
        canopy_parameters=CanopyParams(),
        organic_layer_parameters=OrganicLayerParams(),
        output_parameters=OutputParams(),
        photo_parameters=get_photo_parameters_by_location(
            location=LocationsForPhotoParams("All_data"),
        ),
        site_parameters=SiteParams(
            L=10.0,
            n=n,
            initial_dominant_stand_age_years=70.0,
            initial_subdominant_stand_age_years=70.0,
            initial_understorey_age_years=70.0,
            canopylayers=CanopyLayerAllometryPointers(
                dominant=[1] * n,
                subdominant=[0] * n,
                under=[0] * n,
            ),
            site_fertility_class=4,
            sitename="test",
            species=TreeSpecies("Pine"),
            sfc_specification=1,
            hdom=None,
            vol=None,
            smc="Peatland",
            nLyrs=60,
            dzLyr=0.05,
            ditch_depth_west=[-0.5],
            ditch_depth_east=[-0.5],
            ditch_depth_20y_west=[-0.5],
            ditch_depth_20y_east=[-0.5],
            scenario_name=["test"],
            drain_age=100.0,
            initial_h=-0.2,
            slope=0.0,
            peat_type=[PeatTypes.generic] * 8,
            peat_type_bottom=[PeatTypes.generic],
            anisotropy=10.0,
            vonP=True,
            vonP_top=[2, 5, 5, 5, 6, 6, 7, 7],
            vonP_bottom=8,
            bd_top=None,
            bd_bottom=0.16,
            peatN=None,
            peatP=None,
            peatK=None,
            enable_peattop=True,
            enable_peatmiddle=True,
            enable_peatbottom=True,
            rho_mor=90.0,
            h_mor=h_mor_from_drainage_and_mass_mor_Pitkanen,
            cutting=cutting,
            depoN=4.0,
            depoP=0.1,
            depoK=1.0,
            fertilization=None,
            peat_temperature=PeatTemperatureParams(),
        ),
    )


def _make_regeneration_allometry():
    """Post-clearcut allometry: age must start at 1 (see ClearCutParams's
    new_allometry_includes_age_one validator). test_allometry.xlsx does NOT
    qualify (it starts at age 60) — post_clearcut_allom.xlsx does."""
    return AllometryParams(
        allometry_dir_path=DATA_DIR,
        dominant={1: "post_clearcut_allom.xlsx"},
        subdominant={0: "post_clearcut_allom.xlsx"},
        under={0: "post_clearcut_allom.xlsx"},
    )


class TestCuttingYearBounds:
    """cutting_yr must fall within the simulation period (2004-2007 here).

    NOTE: as of this writing, susi's SusiParams has no equivalent of
    supersusi's `check_cutting_within_years` model_validator (it only has
    `check_fertilization_within_bounds`). Until that validator is ported,
    test_cut_before_start and test_cut_after_end below will fail (no
    ValueError is raised) — that's expected, not a bug in these tests.
    """

    def test_cut_within_bounds(self):
        sp = _make_susi_params(cutting=ThinningParams(cutting_yr=2005, to_ba=12))
        assert sp.site_parameters.cutting.cutting_yr == 2005

    def test_cut_before_start(self):
        with pytest.raises(ValueError, match="out of bounds"):
            _make_susi_params(cutting=ThinningParams(cutting_yr=2003, to_ba=12))

    def test_cut_after_end(self):
        with pytest.raises(ValueError, match="out of bounds"):
            _make_susi_params(cutting=ThinningParams(cutting_yr=2008, to_ba=12))

    def test_cut_on_start_year(self):
        sp = _make_susi_params(cutting=ThinningParams(cutting_yr=2004, to_ba=12))
        assert sp.site_parameters.cutting.cutting_yr == 2004

    def test_cut_on_end_year(self):
        sp = _make_susi_params(cutting=ThinningParams(cutting_yr=2007, to_ba=12))
        assert sp.site_parameters.cutting.cutting_yr == 2007

    def test_no_cut(self):
        sp = _make_susi_params(cutting=None)
        assert sp.site_parameters.cutting is None

    def test_clearcut_within_bounds(self):
        sp = _make_susi_params(
            cutting=ClearCutParams(
                cutting_yr=2006,
                new_growth_allometry=_make_regeneration_allometry(),
                strips_to_cut=[True] * 5,
            ),
        )
        assert sp.site_parameters.cutting.cutting_yr == 2006


class TestClearCutStripsLength:
    """strips_to_cut length must equal the number of soil columns n."""

    def test_clearcut_strips_length_matches(self):
        sp = _make_susi_params(
            cutting=ClearCutParams(
                cutting_yr=2005,
                new_growth_allometry=_make_regeneration_allometry(),
                strips_to_cut=[True] * 5,
            ),
        )
        assert sp.site_parameters.cutting.cutting_yr == 2005

    def test_clearcut_strips_length_mismatch(self):
        with pytest.raises(
            ValueError, match="ClearCutParams.strips_to_cut has 3 elements"
        ):
            _make_susi_params(
                cutting=ClearCutParams(
                    cutting_yr=2005,
                    new_growth_allometry=_make_regeneration_allometry(),
                    strips_to_cut=[True] * 3,
                ),
            )

    def test_thinning_skips_strips_check(self):
        sp = _make_susi_params(cutting=ThinningParams(cutting_yr=2005, to_ba=12))
        assert sp.site_parameters.cutting.cutting_yr == 2005

    def test_no_cut_skips_strips_check(self):
        sp = _make_susi_params(cutting=None)
        assert sp.site_parameters.cutting is None


def _site_params_kwargs(n: int = 5, **overrides):
    """Shared valid SiteParams kwargs, for tests that construct SiteParams
    directly (old-API-rejection happens at SiteParams construction, before
    a cutting=... field ever comes into play)."""
    kwargs = dict(
        L=10.0,
        n=n,
        initial_dominant_stand_age_years=70.0,
        initial_subdominant_stand_age_years=70.0,
        initial_understorey_age_years=70.0,
        canopylayers=CanopyLayerAllometryPointers(
            dominant=[1] * n,
            subdominant=[0] * n,
            under=[0] * n,
        ),
        site_fertility_class=4,
        sitename="test",
        species=TreeSpecies("Pine"),
        sfc_specification=1,
        hdom=None,
        vol=None,
        smc="Peatland",
        nLyrs=60,
        dzLyr=0.05,
        ditch_depth_west=[-0.5],
        ditch_depth_east=[-0.5],
        ditch_depth_20y_west=[-0.5],
        ditch_depth_20y_east=[-0.5],
        scenario_name=["test"],
        drain_age=100.0,
        initial_h=-0.2,
        slope=0.0,
        peat_type=[PeatTypes.generic] * 8,
        peat_type_bottom=[PeatTypes.generic],
        anisotropy=10.0,
        vonP=True,
        vonP_top=[2, 5, 5, 5, 6, 6, 7, 7],
        vonP_bottom=8,
        bd_top=None,
        bd_bottom=0.16,
        peatN=None,
        peatP=None,
        peatK=None,
        enable_peattop=True,
        enable_peatmiddle=True,
        enable_peatbottom=True,
        rho_mor=90.0,
        h_mor=0.04,
        depoN=4.0,
        depoP=0.1,
        depoK=1.0,
        fertilization=None,
        peat_temperature=PeatTemperatureParams(),
    )
    kwargs.update(overrides)
    return kwargs


class TestOldApiRejection:
    """Old flat cutting_yr/cutting_to_ba fields on SiteParams must be rejected
    with a clear message pointing at the new `cutting` field."""

    def test_old_cutting_yr_rejected(self):
        with pytest.raises(ValueError, match="replaced by the `cutting` field"):
            SiteParams(**_site_params_kwargs(cutting_yr=2005))

    def test_old_cutting_to_ba_rejected(self):
        with pytest.raises(ValueError, match="replaced by the `cutting` field"):
            SiteParams(**_site_params_kwargs(cutting_to_ba=12))

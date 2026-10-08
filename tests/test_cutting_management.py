# Model-level tests for the CuttingManagementParams API (issue #171).
#
# Scope: this file only exercises the Pydantic model (SiteParams.cutting_management,
# CuttingManagementParams, ClearCutParams, ThinningParams, and the
# old-API-rejection/strips-length validators). It intentionally does NOT test
# the cutting *behavior* (Canopylayer.cutting() dispatch, strip-selective
# harvesting, post-cut allometry switch-over) — that lives in susi's OOP core
# and hasn't been ported yet. Compare with supersusi's
# tests/tests_cutting_management.py, whose TestClearCutSelection class tests
# exactly that runtime behavior against supersusi's functional
# cut_layer()/apply_allometry(); there is no susi-side equivalent of those
# functions yet, so that class has no counterpart here.
#
# API shape: `cutting_management` on SiteParams holds a `CuttingManagementParams`
# (or None), which carries the shared `application_yr` plus a `management_type`
# of `ClearCutParams | ContinuousCoverParams | ThinningParams`. `cutting_yr` no
# longer exists on ClearCutParams/ThinningParams themselves — it moved up to
# `CuttingManagementParams.application_yr`.
import datetime
from pathlib import Path

import pytest

from susi.io.susi_parameter_model import (
    AllometryFileAndSpecies,
    CanopyLayerAllometry,
    CanopyLayerName,
    CanopyParams,
    ClearCut,
    CuttingManagementParams,
    LocationsForPhotoParams,
    OrganicLayerParams,
    OutputParams,
    PeatTemperatureParams,
    PeatTypes,
    SimulationConfig,
    SiteParams,
    StandParams,
    SusiParams,
    Thinning,
    WeatherParams,
    get_photo_parameters_by_location,
    h_mor_from_drainage_and_mass_mor_Pitkanen,
)

DATA_DIR = Path(__file__).parent / "data"


def _make_susi_params(*, cutting_management, n: int = 5):
    """Build a minimal-but-valid SusiParams with the given cutting intervention.

    Mirrors the `valid_susi_params` fixture in test_susi_model.py, parameterized
    on `cutting_management` so each test only has to specify the piece it cares about.
    """
    return SusiParams(
        weather_parameters=WeatherParams(
            FMI_weather_filepath=DATA_DIR / "weather.csv",
        ),
        simulation_config=SimulationConfig(
            start_date=datetime.datetime(2004, 1, 1),
            end_date=datetime.datetime(2007, 12, 31),
        ),
        stand_params=StandParams(
            site_fertility_class=4,
            canopy_layer_allometry=CanopyLayerAllometry(
                allometry_file_registry={
                    1: AllometryFileAndSpecies(
                        file_path=DATA_DIR / "test_allometry.csv", species_id=1
                    )
                },
                pointers={
                    CanopyLayerName.dominant: [1] * n,
                    CanopyLayerName.subdominant: None,
                    CanopyLayerName.under: None,
                },
            ),
            initial_canopylayer_age_years={
                CanopyLayerName.dominant: 70.0,
                CanopyLayerName.subdominant: 70.0,
                CanopyLayerName.under: 70.0,
            },
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
            sitename="test",
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
            cutting_management=cutting_management,
            depoN=4.0,
            depoP=0.1,
            depoK=1.0,
            fertilization=None,
            peat_temperature=PeatTemperatureParams(),
        ),
    )


def _make_regeneration_allometry(n: int = 5):
    """Post-clearcut allometry: age must start at 1 (see ClearCutParams's
    new_allometry_includes_age_one validator). test_allometry.csv does NOT
    qualify (it starts at age 60) — post_clearcut_allom.csv does."""
    return CanopyLayerAllometry(
        allometry_file_registry={
            1: AllometryFileAndSpecies(
                file_path=DATA_DIR / "post_clearcut_allom.csv", species_id=1
            )
        },
        pointers={
            CanopyLayerName.dominant: [1] * n,
            CanopyLayerName.subdominant: None,
            CanopyLayerName.under: None,
        },
    )


class TestCuttingYearBounds:
    """application_yr must fall within the simulation period (2004-2007 here),
    checked by SusiParams.check_cutting_within_years."""

    def test_cut_within_bounds(self):
        sp = _make_susi_params(
            cutting_management=CuttingManagementParams(
                application_yr=2005,
                management_type=Thinning(
                    target_basal_area={CanopyLayerName.dominant: 12}
                ),
            )
        )
        assert sp.site_parameters.cutting_management.application_yr == 2005

    def test_cut_before_start(self):
        with pytest.raises(ValueError, match="out of bounds"):
            _make_susi_params(
                cutting_management=CuttingManagementParams(
                    application_yr=2003,
                    management_type=Thinning(
                        target_basal_area={CanopyLayerName.dominant: 12}
                    ),
                )
            )

    def test_cut_after_end(self):
        with pytest.raises(ValueError, match="out of bounds"):
            _make_susi_params(
                cutting_management=CuttingManagementParams(
                    application_yr=2008,
                    management_type=Thinning(
                        target_basal_area={CanopyLayerName.dominant: 12}
                    ),
                )
            )

    def test_cut_on_start_year(self):
        sp = _make_susi_params(
            cutting_management=CuttingManagementParams(
                application_yr=2004,
                management_type=Thinning(
                    target_basal_area={CanopyLayerName.dominant: 12}
                ),
            )
        )
        assert sp.site_parameters.cutting_management.application_yr == 2004

    def test_cut_on_end_year(self):
        sp = _make_susi_params(
            cutting_management=CuttingManagementParams(
                application_yr=2007,
                management_type=Thinning(
                    target_basal_area={CanopyLayerName.dominant: 12}
                ),
            )
        )
        assert sp.site_parameters.cutting_management.application_yr == 2007

    def test_no_cut(self):
        sp = _make_susi_params(cutting_management=None)
        assert sp.site_parameters.cutting_management is None

    def test_clearcut_within_bounds(self):
        sp = _make_susi_params(
            cutting_management=CuttingManagementParams(
                application_yr=2006,
                management_type=ClearCut(
                    new_growth_allometry=_make_regeneration_allometry(),
                    strips_to_cut=[True] * 5,
                ),
            ),
        )
        assert sp.site_parameters.cutting_management.application_yr == 2006


class TestClearCutStripsLength:
    """strips_to_cut length must equal the number of soil columns n."""

    def test_clearcut_strips_length_matches(self):
        sp = _make_susi_params(
            cutting_management=CuttingManagementParams(
                application_yr=2005,
                management_type=ClearCut(
                    new_growth_allometry=_make_regeneration_allometry(),
                    strips_to_cut=[True] * 5,
                ),
            ),
        )
        assert sp.site_parameters.cutting_management.application_yr == 2005

    def test_clearcut_strips_length_mismatch(self):
        with pytest.raises(ValueError, match="ClearCut.strips_to_cut has 3 elements"):
            _make_susi_params(
                cutting_management=CuttingManagementParams(
                    application_yr=2005,
                    management_type=ClearCut(
                        new_growth_allometry=_make_regeneration_allometry(n=3),
                        strips_to_cut=[True] * 3,
                    ),
                ),
            )

    def test_thinning_skips_strips_check(self):
        sp = _make_susi_params(
            cutting_management=CuttingManagementParams(
                application_yr=2005,
                management_type=Thinning(
                    target_basal_area={CanopyLayerName.dominant: 12}
                ),
            )
        )
        assert sp.site_parameters.cutting_management.application_yr == 2005

    def test_no_cut_skips_strips_check(self):
        sp = _make_susi_params(cutting_management=None)
        assert sp.site_parameters.cutting_management is None


class TestThinningTargetsExistingLayers:
    """Thinning.target_basal_area must only name layers that have allometry
    (non-None pointers in stand_params.canopy_layer_allometry) -- see
    SusiParams.thinning_only_targets_layers_with_allometry. In
    _make_susi_params, only the 'dominant' layer has allometry; subdominant
    and under are both None."""

    def test_thinning_existing_layer(self):
        sp = _make_susi_params(
            cutting_management=CuttingManagementParams(
                application_yr=2005,
                management_type=Thinning(
                    target_basal_area={CanopyLayerName.dominant: 12}
                ),
            )
        )
        assert sp.site_parameters.cutting_management.application_yr == 2005

    def test_thinning_nonexistent_layer_rejected(self):
        with pytest.raises(
            ValueError, match="'subdominant' layer, but that layer does not exist"
        ):
            _make_susi_params(
                cutting_management=CuttingManagementParams(
                    application_yr=2005,
                    management_type=Thinning(
                        target_basal_area={CanopyLayerName.subdominant: 12}
                    ),
                )
            )

    def test_thinning_mix_of_existing_and_nonexistent_layer_rejected(self):
        with pytest.raises(
            ValueError, match="'under' layer, but that layer does not exist"
        ):
            _make_susi_params(
                cutting_management=CuttingManagementParams(
                    application_yr=2005,
                    management_type=Thinning(
                        target_basal_area={
                            CanopyLayerName.dominant: 12,
                            CanopyLayerName.under: 8,
                        }
                    ),
                )
            )

    def test_no_cut_skips_existing_layer_check(self):
        sp = _make_susi_params(cutting_management=None)
        assert sp.site_parameters.cutting_management is None


def _site_params_kwargs(n: int = 5, **overrides):
    """Shared valid SiteParams kwargs, for tests that construct SiteParams
    directly (old-API-rejection happens at SiteParams construction, before
    a cutting_management=... field ever comes into play)."""
    kwargs = {
        "L": 10.0,
        "n": n,
        "sitename": "test",
        "sfc_specification": 1,
        "hdom": None,
        "vol": None,
        "smc": "Peatland",
        "nLyrs": 60,
        "dzLyr": 0.05,
        "ditch_depth_west": [-0.5],
        "ditch_depth_east": [-0.5],
        "ditch_depth_20y_west": [-0.5],
        "ditch_depth_20y_east": [-0.5],
        "scenario_name": ["test"],
        "drain_age": 100.0,
        "initial_h": -0.2,
        "slope": 0.0,
        "peat_type": [PeatTypes.generic] * 8,
        "peat_type_bottom": [PeatTypes.generic],
        "anisotropy": 10.0,
        "vonP": True,
        "vonP_top": [2, 5, 5, 5, 6, 6, 7, 7],
        "vonP_bottom": 8,
        "bd_top": None,
        "bd_bottom": 0.16,
        "peatN": None,
        "peatP": None,
        "peatK": None,
        "enable_peattop": True,
        "enable_peatmiddle": True,
        "enable_peatbottom": True,
        "rho_mor": 90.0,
        "h_mor": 0.04,
        "depoN": 4.0,
        "depoP": 0.1,
        "depoK": 1.0,
        "fertilization": None,
        "peat_temperature": PeatTemperatureParams(),
    }
    kwargs.update(overrides)
    return kwargs


class TestOldApiRejection:
    """Old flat cutting_yr/cutting_to_ba fields on SiteParams (pre-#171, v0 API)
    must be rejected with a clear message pointing at the new `cutting_management`
    field. Unaffected by the CuttingManagementParams wrapper rename — this
    validator only inspects the raw construction kwargs for the long-gone v0
    field names."""

    def test_old_cutting_yr_rejected(self):
        with pytest.raises(
            ValueError, match="replaced by the `cutting_management` field"
        ):
            SiteParams(**_site_params_kwargs(cutting_yr=2005))

    def test_old_cutting_to_ba_rejected(self):
        with pytest.raises(
            ValueError, match="replaced by the `cutting_management` field"
        ):
            SiteParams(**_site_params_kwargs(cutting_to_ba=12))

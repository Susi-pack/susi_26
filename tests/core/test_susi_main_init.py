import datetime

import numpy as np
import pytest

from supersusi.io.susi_parameter_model import (
    SusiParams, SiteParams, WeatherParams, SimulationConfig,
    AllometryParams, CanopyParams, OrganicLayerParams, OutputParams,
    PeatTemperatureParams,
    PeatTypes, TreeSpecies,
    get_photo_parameters_by_location, LocationsForPhotoParams,
    h_mor_from_drainage_and_mass_mor_Pitkanen,
    CanopyLayerAllometryPointers,
)
from supersusi.io.app_settings import AppSettings

from supersusi.core.susi_main import (
    _build_params, _compute_constants, _init_state,
    AllState, DailyState, AnnualState,
)


_app_settings = AppSettings()
TEST_DATA = _app_settings.input_folder


@pytest.fixture
def params_and_constants():
    """Minimal SusiParams + weather_data → ModuleParams + ModuleComputedConstants."""
    weather_file = TEST_DATA / "weather" / "CFw.csv"

    sp = SusiParams(
        weather_parameters=WeatherParams(FMI_weather_filepath=weather_file),
        simulation_config=SimulationConfig(
            start_date=datetime.datetime(2004, 1, 1),
            end_date=datetime.datetime(2004, 12, 31),
        ),
        allometry_parameters=AllometryParams(
            allometry_dir_path=TEST_DATA,
            dominant={1: "CF_41.xlsx"},
            subdominant={0: "susi_motti_input_lyr_1.xlsx"},
            under={0: "susi_motti_input_lyr_2.xlsx"},
        ),
        canopy_parameters=CanopyParams(),
        organic_layer_parameters=OrganicLayerParams(),
        output_parameters=OutputParams(),
        photo_parameters=get_photo_parameters_by_location(
            location=LocationsForPhotoParams("All_data")
        ),
        site_parameters=SiteParams(
            L=40.0, n=10,
            initial_dominant_stand_age_years=60.0,
            initial_subdominant_stand_age_years=0.0,
            initial_understorey_age_years=0.0,
            canopylayers=CanopyLayerAllometryPointers(
                dominant=[1] * 10, subdominant=[0] * 10, under=[0] * 10
            ),
            site_fertility_class=4, sitename="test",
            species=TreeSpecies("Pine"), sfc_specification=1,
            hdom=None, vol=None, smc="Peatland",
            nLyrs=60, dzLyr=0.05,
            ditch_depth_west=[-0.5], ditch_depth_east=[-0.5],
            ditch_depth_20y_west=[-0.5], ditch_depth_20y_east=[-0.5],
            scenario_name=["test"], drain_age=100.0,
            initial_h=-0.2, slope=0.0,
            peat_type=[PeatTypes.generic] * 8,
            peat_type_bottom=[PeatTypes.generic],
            anisotropy=10.0, vonP=True,
            vonP_top=[2, 5, 5, 5, 6, 6, 7, 7], vonP_bottom=8,
            bd_top=None, bd_bottom=0.16,
            peatN=None, peatP=None, peatK=None,
            enable_peattop=True, enable_peatmiddle=True, enable_peatbottom=True,
            rho_mor=90.0, h_mor=h_mor_from_drainage_and_mass_mor_Pitkanen,
            cutting_yr=2004, cutting_to_ba=12,
            depoN=4.0, depoP=0.1, depoK=1.0,
            fertilization=None,  # no fertilization for test
            peat_temperature=PeatTemperatureParams(),
        ),
    )

    from supersusi.io.forcing_weather import read_FMI_weather
    weather_data = read_FMI_weather(
        ID=0,
        start_date=sp.simulation_config.start_date,
        end_date=sp.simulation_config.end_date,
        sourcefile=sp.weather_parameters.FMI_weather_filepath,
    )

    params = _build_params(sp)
    constants = _compute_constants(params, sp, weather_data)
    return params, constants, sp, weather_data


class TestInitState:
    def test_returns_all_state(self, params_and_constants):
        params, constants, sp, _ = params_and_constants
        state, *_ = _init_state(params, constants, sp)
        assert isinstance(state, AllState)
        assert isinstance(state.daily, DailyState)
        assert isinstance(state.annual, AnnualState)

    def test_all_fields_populated(self, params_and_constants):
        params, constants, sp, _ = params_and_constants
        state, *_ = _init_state(params, constants, sp)
        n = sp.site_parameters.n

        # Daily
        assert state.daily.canopy is not None
        assert state.daily.moss is not None
        assert state.daily.strip is not None
        assert state.daily.peat_T is not None
        assert state.daily.canopy.W.shape == (n,)

        # Annual
        assert state.annual.stand is not None
        assert state.annual.stand_outputs is not None
        assert state.annual.gv is not None
        assert state.annual.esom_mass is not None
        assert state.annual.esom_N is not None
        assert state.annual.esom_P is not None
        assert state.annual.esom_K is not None
        assert state.annual.stand_outputs.volume.shape == (n,)

    def test_canopy_amax_updated(self, params_and_constants):
        """When nut_stat != 1, update_amax should change canopy.amax."""
        params, constants, sp, _ = params_and_constants
        state, *_ = _init_state(params, constants, sp)

        # nut_stat defaults to ones(10), so amax should equal amax_init
        from supersusi.core.canopygrid import compute_initial_state, update_amax
        raw = compute_initial_state(params.canopygrid)
        expected = update_amax(state.annual.stand.nut_stat, raw)
        np.testing.assert_array_equal(state.daily.canopy.amax, expected.amax)

    def test_module_states_match_direct_calls(self, params_and_constants):
        """Each module state equals calling its compute_initial_state directly."""
        params, constants, sp, _ = params_and_constants
        state, *_ = _init_state(params, constants, sp)

        from supersusi.core import canopygrid, mosslayer, temperature, strip, gvegetation, esom

        # canopy
        raw = canopygrid.compute_initial_state(params.canopygrid)
        expected_cpy = canopygrid.update_amax(state.annual.stand.nut_stat, raw)
        np.testing.assert_array_equal(state.daily.canopy.amax, expected_cpy.amax)

        # moss
        expected_moss = mosslayer.compute_initial_state(params.mosslayer, constants.mosslayer)
        np.testing.assert_array_equal(state.daily.moss.Wsto_top, expected_moss.Wsto_top)

        # temperature
        expected_T = temperature.compute_initial_state(constants.temperature)
        np.testing.assert_array_equal(state.daily.peat_T.T_soil, expected_T.T_soil)

        # strip
        expected_strip = strip.compute_initial_state(params.strip, constants.strip)
        np.testing.assert_array_equal(state.daily.strip.H, expected_strip.H)

        # gv
        expected_gv = gvegetation.compute_initial_state(params.gvegetation, constants.gvegetation)
        np.testing.assert_array_equal(state.annual.gv.gv_tot, expected_gv.gv_tot)

        # esom
        esom_map = {"esom_mass": "mass", "esom_N": "n", "esom_P": "p", "esom_K": "k"}
        for attr, sub in esom_map.items():
            got = getattr(state.annual, attr)
            expected = esom.compute_initial_state(
                getattr(params.esom, sub),
                getattr(constants.esom, sub),
            )
            np.testing.assert_array_equal(got.M, expected.M)
            assert got.i == expected.i

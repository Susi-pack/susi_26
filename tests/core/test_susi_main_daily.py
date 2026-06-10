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
    _run_daily_step,
    DailyForcing,
)


_app_settings = AppSettings()
TEST_DATA = _app_settings.input_folder


@pytest.fixture(scope="module")
def setup():
    """Build params, constants, and initial state for daily-step tests."""
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
                dominant=[1] * 10, subdominant=[0] * 10, under=[0] * 10,
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
            fertilization=None,
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
    state, *_ = _init_state(params, constants, sp)

    from supersusi.core import strip as strip_mod
    buffer = strip_mod.make_numerical_buffer(params.strip)

    return {
        "params": params,
        "constants": constants,
        "init_state": state,
        "weather": weather_data,
        "buffer": buffer,
        "sp": sp,
    }


def _first_forcing(setup):
    w = setup["weather"].iloc[0]
    return DailyForcing(
        T=w["T"], Prec=w["Prec"], Rg=w["Rg"], Par=w["Par"], VPD=w["vpd"],
        h0ts_west=-0.5, h0ts_east=-0.5,
    )


class TestRunDailyStep:
    def test_returns_correct_types(self, setup):
        s = setup
        daily = s["init_state"].daily
        forcing = _first_forcing(setup)
        new_daily, dout = _run_daily_step(
            daily, forcing,
            hdom=s["init_state"].annual.stand_outputs.hdom,
            leafarea=s["init_state"].annual.stand_outputs.leafarea,
            params=s["params"], constants=s["constants"],
            buffer=s["buffer"],
        )
        assert isinstance(new_daily, type(daily))
        assert new_daily is not daily

    def test_all_outputs_populated(self, setup):
        s = setup
        n = s["sp"].site_parameters.n
        n_hydro = s["sp"].site_parameters.nLyrs
        daily = s["init_state"].daily
        forcing = _first_forcing(setup)
        _, dout = _run_daily_step(
            daily, forcing,
            hdom=s["init_state"].annual.stand_outputs.hdom,
            leafarea=s["init_state"].annual.stand_outputs.leafarea,
            params=s["params"], constants=s["constants"],
            buffer=s["buffer"],
        )
        assert dout.wtd.shape == (n,)
        assert dout.afp.shape == (n,)
        assert dout.T_soil_hydro.shape == (n_hydro,)
        assert dout.delta.shape == (n,)
        assert isinstance(dout.total_runoff, np.ndarray)
        assert dout.surface_runoff.shape == (n,)
        assert dout.interc.shape == (n,)
        assert dout.evap.shape == (n,)
        assert dout.et.shape == (n,)
        assert dout.transpi.shape == (n,)
        assert dout.efloor.shape == (n,)
        assert dout.swe.shape == (n,)
        assert not np.any(np.isnan(dout.wtd))
        assert not np.any(np.isnan(dout.afp))
        assert not np.any(np.isnan(dout.delta))

    def test_daily_state_updated(self, setup):
        s = setup
        daily = s["init_state"].daily
        forcing = _first_forcing(setup)
        new_daily, _ = _run_daily_step(
            daily, forcing,
            hdom=s["init_state"].annual.stand_outputs.hdom,
            leafarea=s["init_state"].annual.stand_outputs.leafarea,
            params=s["params"], constants=s["constants"],
            buffer=s["buffer"],
        )
        assert new_daily.canopy is not daily.canopy
        assert new_daily.moss is not daily.moss
        assert new_daily.strip is not daily.strip
        assert new_daily.peat_T is not daily.peat_T

    def test_efloor_pre_moss_in_output(self, setup):
        """DailyOutputs.efloor should be pre-moss canopy efloor,
        while temperature receives post-moss efloor (moss_interc_out.evap)."""
        s = setup
        daily = s["init_state"].daily
        forcing = _first_forcing(setup)

        # Run canopy step in isolation to get pre-moss efloor
        from supersusi.core.susi_utils import rew_drylimit
        from supersusi.io.forcing_weather import WeatherForcings
        from supersusi.core import canopygrid

        n = s["sp"].site_parameters.n
        hdom = s["init_state"].annual.stand_outputs.hdom
        leafarea = s["init_state"].annual.stand_outputs.leafarea
        rew = rew_drylimit(s["params"].strip.initial_h * np.ones(n))
        cpy_inputs = canopygrid.assemble_inputs(
            WeatherForcings(T=forcing.T, Prec=forcing.Prec,
                            Rg=forcing.Rg, Par=forcing.Par, VPD=forcing.VPD),
            hc=hdom, LAIconif=leafarea, Rew=rew, beta=daily.moss.Ree,
        )
        _, cpy_out = canopygrid.run_timestep(
            s["params"].canopygrid, cpy_inputs, daily.canopy,
        )

        _, dout = _run_daily_step(
            daily, forcing, hdom=hdom, leafarea=leafarea,
            params=s["params"], constants=s["constants"],
            buffer=s["buffer"],
        )

        # efloor in DailyOutputs = pre-moss canopy efloor
        np.testing.assert_array_equal(dout.efloor, cpy_out.efloor)

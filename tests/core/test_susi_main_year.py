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
    _run_annual_step,
    AnnualForcing, AllState,
)


_app_settings = AppSettings()
TEST_DATA = _app_settings.input_folder


@pytest.fixture(scope="module")
def setup():
    """Build params, constants, state, and a single AnnualForcing."""
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

    # Manually build AnnualForcing for 2004
    ndays = 366  # 2004 is a leap year
    cal_year = 2004
    w_yr = weather_data.iloc[:ndays]

    year_forcing = AnnualForcing(
        daily_T=w_yr["T"].values.astype(float),
        daily_Rg=w_yr["Rg"].values.astype(float),
        daily_VPD=w_yr["vpd"].values.astype(float),
        daily_Prec=w_yr["Prec"].values.astype(float),
        daily_Par=w_yr["Par"].values.astype(float),
        h0ts_west=np.full(ndays, -0.5, dtype=float),
        h0ts_east=np.full(ndays, -0.5, dtype=float),
        valid_days=ndays,
        calendar_year=cal_year,
        temp_sum=constants.age * 0.0,  # dummy
        do_cutting=True,
        cutting_to_ba=12.0,
    )

    return {
        "params": params,
        "constants": constants,
        "init_state": state,
        "weather_data": weather_data,
        "buffer": buffer,
        "sp": sp,
        "year_forcing": year_forcing,
    }


class TestRunAnnualStep:
    def test_returns_correct_types(self, setup):
        s = setup
        state, out = _run_annual_step(
            s["init_state"], s["year_forcing"], s["weather_data"],
            s["params"], s["constants"], s["buffer"],
        )
        assert isinstance(state, AllState)
        assert isinstance(state.daily.canopy, object)
        # AnnualOutputs imported locally to avoid circular issues
        from supersusi.core.susi_main import AnnualOutputs
        assert isinstance(out, AnnualOutputs)

    def test_daily_output_shape(self, setup):
        s = setup
        ndays = s["year_forcing"].valid_days
        n = s["sp"].site_parameters.n
        n_hydro = s["sp"].site_parameters.nLyrs
        state, out = _run_annual_step(
            s["init_state"], s["year_forcing"], s["weather_data"],
            s["params"], s["constants"], s["buffer"],
        )
        assert out.daily.wtd.shape == (ndays, n)
        assert out.daily.afp.shape == (ndays, n)
        assert out.daily.T_soil_hydro.shape == (ndays, n_hydro)
        assert out.daily.delta.shape == (ndays, n)

    def test_annual_outputs_populated(self, setup):
        s = setup
        n = s["sp"].site_parameters.n
        state, out = _run_annual_step(
            s["init_state"], s["year_forcing"], s["weather_data"],
            s["params"], s["constants"], s["buffer"],
        )
        assert out.stand.volume.shape == (n,)
        assert out.gv.gv_field.shape == (n,)
        assert out.gv.gv_bot.shape == (n,)
        assert out.gv.gv_change.shape == (n,)
        assert out.esom_mass.out.shape == (n,)
        assert out.esom_N.out.shape == (n,)
        assert out.esom_P.out.shape == (n,)
        assert out.esom_K.out.shape == (n,)
        assert out.methane.ch4.shape == (n,)
        assert out.Rhet.shape == (n,)
        assert out.soil_co2_balance.shape == (n,)
        assert not np.any(np.isnan(out.stand.volume))
        assert not np.any(np.isnan(out.gv.gv_field))
        assert not np.any(np.isnan(out.methane.ch4))

    def test_daily_state_updated(self, setup):
        s = setup
        state, out = _run_annual_step(
            s["init_state"], s["year_forcing"], s["weather_data"],
            s["params"], s["constants"], s["buffer"],
        )
        assert state.daily.canopy is not s["init_state"].daily.canopy
        assert state.daily.strip is not s["init_state"].daily.strip
        assert state.daily.peat_T is not s["init_state"].daily.peat_T

    def test_annual_state_updated(self, setup):
        s = setup
        state, out = _run_annual_step(
            s["init_state"], s["year_forcing"], s["weather_data"],
            s["params"], s["constants"], s["buffer"],
        )
        assert state.annual.stand is not s["init_state"].annual.stand
        assert state.annual.gv is not s["init_state"].annual.gv
        assert state.annual.esom_mass is not s["init_state"].annual.esom_mass
        assert state.annual.esom_N is not s["init_state"].annual.esom_N
        assert state.annual.esom_P is not s["init_state"].annual.esom_P
        assert state.annual.esom_K is not s["init_state"].annual.esom_K

    def test_cutting_applied(self, setup):
        """With do_cutting=True, stand outputs should reflect cutting."""
        s = setup
        state_no_cut = s["init_state"]
        # Create forcing with do_cutting=False
        yr = s["year_forcing"]
        yr_no_cut = AnnualForcing(
            daily_T=yr.daily_T, daily_Rg=yr.daily_Rg,
            daily_VPD=yr.daily_VPD, daily_Prec=yr.daily_Prec,
            daily_Par=yr.daily_Par,
            h0ts_west=yr.h0ts_west, h0ts_east=yr.h0ts_east,
            valid_days=yr.valid_days, calendar_year=yr.calendar_year,
            temp_sum=yr.temp_sum,
            do_cutting=False, cutting_to_ba=0.0,
        )
        cut_state, cut_out = _run_annual_step(
            s["init_state"], yr, s["weather_data"],
            s["params"], s["constants"], s["buffer"],
        )
        no_cut_state, no_cut_out = _run_annual_step(
            state_no_cut, yr_no_cut, s["weather_data"],
            s["params"], s["constants"], s["buffer"],
        )
        # Cutting reduces volume
        assert np.all(cut_out.stand.volume <= no_cut_out.stand.volume + 1e-10)

    def test_nonzero_outputs(self, setup):
        """After a year, some outputs should be non-zero."""
        s = setup
        state, out = _run_annual_step(
            s["init_state"], s["year_forcing"], s["weather_data"],
            s["params"], s["constants"], s["buffer"],
        )
        assert np.any(out.gv.gv_field > 0)
        assert np.any(out.methane.ch4 != 0)

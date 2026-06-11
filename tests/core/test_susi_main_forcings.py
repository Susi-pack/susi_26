import datetime

import pytest

from supersusi.io.susi_parameter_model import (
    SusiParams,
    SiteParams,
    WeatherParams,
    SimulationConfig,
    AllometryParams,
    CanopyParams,
    OrganicLayerParams,
    OutputParams,
    PeatTemperatureParams,
    PeatTypes,
    TreeSpecies,
    get_photo_parameters_by_location,
    LocationsForPhotoParams,
    h_mor_from_drainage_and_mass_mor_Pitkanen,
    CanopyLayerAllometryPointers,
)
from supersusi.io.app_settings import AppSettings

from supersusi.core.susi_main import (
    _build_params,
    _build_annual_forcings,
    AnnualForcing,
)


_app_settings = AppSettings()
TEST_DATA = _app_settings.input_folder


@pytest.fixture(scope="module")
def setup():
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
            L=40.0,
            n=10,
            initial_dominant_stand_age_years=60.0,
            initial_subdominant_stand_age_years=0.0,
            initial_understorey_age_years=0.0,
            canopylayers=CanopyLayerAllometryPointers(
                dominant=[1] * 10,
                subdominant=[0] * 10,
                under=[0] * 10,
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
            cutting_yr=2004,
            cutting_to_ba=12,
            depoN=4.0,
            depoP=0.1,
            depoK=1.0,
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

    return {"sp": sp, "weather_data": weather_data, "params": params}


class TestBuildAnnualForcings:
    def test_returns_list(self, setup):
        forcings = _build_annual_forcings(
            setup["sp"],
            setup["weather_data"],
        )
        assert isinstance(forcings, list)
        assert len(forcings) > 0

    def test_single_year(self, setup):
        forcings = _build_annual_forcings(
            setup["sp"],
            setup["weather_data"],
        )
        assert len(forcings) == 1

    def test_returns_annual_forcing(self, setup):
        forcings = _build_annual_forcings(
            setup["sp"],
            setup["weather_data"],
        )
        assert isinstance(forcings[0], AnnualForcing)

    def test_correct_daily_shapes(self, setup):
        sp = setup["sp"]
        forcings = _build_annual_forcings(
            sp,
            setup["weather_data"],
        )
        yr = forcings[0]
        ndays = yr.valid_days
        assert yr.daily_T.shape == (ndays,)
        assert yr.daily_Rg.shape == (ndays,)
        assert yr.daily_VPD.shape == (ndays,)
        assert yr.daily_Prec.shape == (ndays,)
        assert yr.daily_Par.shape == (ndays,)
        assert yr.h0ts_west.shape == (ndays,)
        assert yr.h0ts_east.shape == (ndays,)

    def test_temp_sum_shape(self, setup):
        sp = setup["sp"]
        forcings = _build_annual_forcings(
            sp,
            setup["weather_data"],
        )
        assert forcings[0].temp_sum.shape == (sp.site_parameters.n,)

    def test_cutting_flag(self, setup):
        forcings = _build_annual_forcings(
            setup["sp"],
            setup["weather_data"],
        )
        assert forcings[0].do_cutting is True
        assert forcings[0].cutting_to_ba == 12.0

    def test_calendar_year(self, setup):
        forcings = _build_annual_forcings(
            setup["sp"],
            setup["weather_data"],
        )
        assert forcings[0].calendar_year == 2004

    def test_multi_year(self, setup):
        sp_multi = SusiParams(
            weather_parameters=setup["sp"].weather_parameters,
            simulation_config=SimulationConfig(
                start_date=datetime.datetime(2004, 1, 1),
                end_date=datetime.datetime(2005, 12, 31),
            ),
            allometry_parameters=setup["sp"].allometry_parameters,
            canopy_parameters=setup["sp"].canopy_parameters,
            organic_layer_parameters=setup["sp"].organic_layer_parameters,
            output_parameters=setup["sp"].output_parameters,
            photo_parameters=setup["sp"].photo_parameters,
            site_parameters=setup["sp"].site_parameters,
        )
        from supersusi.io.forcing_weather import read_FMI_weather

        wd2 = read_FMI_weather(
            ID=0,
            start_date=sp_multi.simulation_config.start_date,
            end_date=sp_multi.simulation_config.end_date,
            sourcefile=setup["sp"].weather_parameters.FMI_weather_filepath,
        )
        forcings = _build_annual_forcings(sp_multi, wd2)
        assert len(forcings) == 2
        assert forcings[0].calendar_year == 2004
        assert forcings[1].calendar_year == 2005
        assert forcings[0].valid_days == 366  # 2004 is leap
        assert forcings[1].valid_days == 365  # 2005 is not

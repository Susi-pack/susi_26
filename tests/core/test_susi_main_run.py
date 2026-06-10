import datetime
import tempfile
from pathlib import Path

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
from supersusi.io.execution_config import SimulationParams
from supersusi.io.metadata_model import SimulationMetaData

from supersusi.core.susi_main import (
    run,
    SimulationOutput, AllState, AnnualOutputs,
)


_app_settings = AppSettings()
TEST_DATA = _app_settings.input_folder


@pytest.fixture(scope="module")
def setup():
    """Build params and simulation params once."""
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
            location=LocationsForPhotoParams("All_data"),
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

    tmpdir = Path(tempfile.mkdtemp(prefix="susi_test_"))
    metadata = SimulationMetaData(
        experiment_id="test_run",
        parent_output_folder=tmpdir,
    )
    sim_params = SimulationParams(susi_params=sp, metadata=metadata)
    result = run(sim_params)
    yield result
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestRunSimulation:
    def test_returns_simulation_output(self, setup):
        assert isinstance(setup, SimulationOutput)

    def test_has_one_year(self, setup):
        assert isinstance(setup.annual, list)
        assert len(setup.annual) == 1

    def test_annual_output_type(self, setup):
        assert isinstance(setup.annual[0], AnnualOutputs)

    def test_final_state_type(self, setup):
        assert isinstance(setup.final_state, AllState)

    def test_outputs_populated(self, setup):
        ann = setup.annual[0]
        n = 10
        assert ann.stand.volume.shape == (n,)
        assert ann.gv.gv_field.shape == (n,)
        assert ann.methane.ch4.shape == (n,)
        assert np.any(ann.gv.gv_field > 0)

    def test_daily_output_shape(self, setup):
        ann = setup.annual[0]
        n = 10
        ndays = ann.daily.wtd.shape[0]
        assert ndays == 366
        assert ann.daily.wtd.shape == (ndays, n)
        assert ann.daily.T_soil_hydro.shape == (ndays, 60)

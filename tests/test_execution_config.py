import datetime
import pytest
from pathlib import Path
from pydantic import ValidationError

from susi.io.execution_config import MultipleSusis, SimulationParams
from susi.io.metadata_model import SimulationMetaData
from susi.io.susi_parameter_model import (
    SusiParams,
    WeatherParams,
    SimulationConfig,
    CanopyLayerAllometry,
    CanopyLayerName,
    CanopyParams,
    OrganicLayerParams,
    OutputParams,
    SiteParams,
    PeatTypes,
    TreeSpecies,
    get_photo_parameters_by_location,
    LocationsForPhotoParams,
    h_mor_from_drainage_and_mass_mor_Pitkanen,
    StandardNPKFertilizationParameters,
    NutrientFertilizationParameters,
    PeatTemperatureParams,
    Thinning,
    CuttingManagementParams,
)


@pytest.fixture
def test_data_path():
    return Path(__file__).parent / "data"


@pytest.fixture
def valid_susi_params(test_data_path):
    return SusiParams(
        weather_parameters=WeatherParams(
            FMI_weather_filepath=test_data_path / "weather.csv",
        ),
        simulation_config=SimulationConfig(
            start_date=datetime.datetime(2004, 1, 1),
            end_date=datetime.datetime(2007, 12, 31),
        ),
        allometry_parameters=CanopyLayerAllometry(
            allometry_dir_path=test_data_path,
            allometry_file_registry={1: "test_allometry.xlsx"},
            pointers={
                CanopyLayerName.dominant: [1, 1, 1, 1, 1],
                CanopyLayerName.subdominant: None,
                CanopyLayerName.under: None,
            },
        ),
        canopy_parameters=CanopyParams(),
        organic_layer_parameters=OrganicLayerParams(),
        output_parameters=OutputParams(),
        photo_parameters=get_photo_parameters_by_location(
            location=LocationsForPhotoParams("All_data")
        ),
        site_parameters=SiteParams(
            L=10.0,
            n=5,
            initial_canopylayer_age_years={
                CanopyLayerName.dominant: 70.0,
                CanopyLayerName.subdominant: 70.0,
                CanopyLayerName.under: 70.0,
            },
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
            cutting_management=CuttingManagementParams(
                application_yr=2004, management_type=Thinning(to_ba=12)
            ),
            depoN=4.0,
            depoP=0.1,
            depoK=1.0,
            fertilization=StandardNPKFertilizationParameters(
                application_year=2005,
                N=NutrientFertilizationParameters(
                    dose=0.0,
                    decay_k=0.5,
                    eff=1.0,
                ),
                P=NutrientFertilizationParameters(dose=45.0, decay_k=0.2, eff=1.0),
                K=NutrientFertilizationParameters(dose=100.0, decay_k=0.3, eff=1.0),
                pH_increment=1.0,
            ),
            peat_temperature=PeatTemperatureParams(),
        ),
    )


@pytest.fixture
def another_valid_susi_params(test_data_path):
    return SusiParams(
        weather_parameters=WeatherParams(
            FMI_weather_filepath=test_data_path / "weather.csv",
        ),
        simulation_config=SimulationConfig(
            start_date=datetime.datetime(2004, 1, 1),
            end_date=datetime.datetime(2007, 12, 31),
        ),
        allometry_parameters=CanopyLayerAllometry(
            allometry_dir_path=test_data_path,
            allometry_file_registry={1: "test_allometry.xlsx"},
            pointers={
                CanopyLayerName.dominant: [1, 1, 1, 1, 1],
                CanopyLayerName.subdominant: None,
                CanopyLayerName.under: None,
            },
        ),
        canopy_parameters=CanopyParams(),
        organic_layer_parameters=OrganicLayerParams(),
        output_parameters=OutputParams(),
        photo_parameters=get_photo_parameters_by_location(
            location=LocationsForPhotoParams("All_data")
        ),
        site_parameters=SiteParams(
            L=10.0,
            n=5,
            initial_canopylayer_age_years={
                CanopyLayerName.dominant: 70.0,
                CanopyLayerName.subdominant: 70.0,
                CanopyLayerName.under: 70.0,
            },
            site_fertility_class=4,
            sitename="test2",
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
            cutting_management=CuttingManagementParams(
                application_yr=2004, management_type=Thinning(to_ba=12)
            ),
            depoN=4.0,
            depoP=0.1,
            depoK=1.0,
            fertilization=StandardNPKFertilizationParameters(
                application_year=2005,
                N=NutrientFertilizationParameters(
                    dose=0.0,
                    decay_k=0.5,
                    eff=1.0,
                ),
                P=NutrientFertilizationParameters(dose=45.0, decay_k=0.2, eff=1.0),
                K=NutrientFertilizationParameters(dose=100.0, decay_k=0.3, eff=1.0),
                pH_increment=1.0,
            ),
            peat_temperature=PeatTemperatureParams(),
        ),
    )


@pytest.fixture
def two_duplicate_susi_params(valid_susi_params) -> list[SimulationParams]:
    return [
        SimulationParams(
            susi_params=valid_susi_params,
            metadata=SimulationMetaData(experiment_id=str(i)),
        )
        for i in range(2)
    ]


@pytest.fixture
def two_duplicate_experiment_folder_paths(
    valid_susi_params, another_valid_susi_params
) -> list[SimulationParams]:
    return [
        SimulationParams(
            susi_params=valid_susi_params,
            metadata=SimulationMetaData(experiment_id=str("THE_SAME")),
        ),
        SimulationParams(
            susi_params=another_valid_susi_params,
            metadata=SimulationMetaData(experiment_id=str("THE_SAME")),
        ),
    ]


@pytest.fixture
def two_valid_simus(
    valid_susi_params, another_valid_susi_params
) -> list[SimulationParams]:
    return [
        SimulationParams(
            susi_params=valid_susi_params,
            metadata=SimulationMetaData(
                experiment_id="one",
                stand_id="stand_A",
                scenario_id="scenario_1",
            ),
        ),
        SimulationParams(
            susi_params=another_valid_susi_params,
            metadata=SimulationMetaData(
                experiment_id="two",
                stand_id="stand_B",
                scenario_id="scenario_1",
            ),
        ),
    ]


@pytest.fixture
def one_hundred_valid_simus(valid_susi_params) -> list[SimulationParams]:
    return [
        SimulationParams(
            susi_params=valid_susi_params.model_copy(
                update={"params_schema_version": i}
            ),
            metadata=SimulationMetaData(
                experiment_id=str(i),
                stand_id=f"stand_{i}",
                scenario_id="scenario_1",
            ),
        )
        for i in range(100)
    ]


def test_duplicate_susi_params(two_duplicate_susi_params):
    """
    Computing the same twice would not make sense
    Make sure there are no duplicated simulation parameters
    """
    with pytest.raises(ValueError):
        MultipleSusis(
            n_parallel_processes=1,
            simulation_parameter_list=two_duplicate_susi_params,
        )


def test_duplicate_folder_names(two_duplicate_experiment_folder_paths):
    """
    Storing Susi results twice in the same folder
    would rewrite the previous contents of the folder
    """
    with pytest.raises(ValueError):
        MultipleSusis(
            n_parallel_processes=1,
            simulation_parameter_list=two_duplicate_experiment_folder_paths,
        )


def test_maximum_number_of_parallel_processes_validation(one_hundred_valid_simus):
    # No more than 40 cores are allowed in multiprocessing
    with pytest.raises(ValueError):
        MultipleSusis(
            n_parallel_processes=41,
            simulation_parameter_list=one_hundred_valid_simus,
        )


def test_less_parallel_processes_than_simus(two_valid_simus):
    with pytest.raises(ValueError):
        MultipleSusis(
            n_parallel_processes=3,
            simulation_parameter_list=two_valid_simus,
        )


def test_valid_batch_multiple_stands_scenarios(
    valid_susi_params, another_valid_susi_params
):
    """Valid batch run with multiple unique stand/scenario combos passes validation."""
    simus = [
        SimulationParams(
            susi_params=valid_susi_params,
            metadata=SimulationMetaData(
                experiment_id="batch_exp",
                stand_id="stand_A",
                scenario_id="scenario_1",
            ),
        ),
        SimulationParams(
            susi_params=another_valid_susi_params,
            metadata=SimulationMetaData(
                experiment_id="batch_exp",
                stand_id="stand_A",
                scenario_id="scenario_2",
            ),
        ),
    ]
    multiple = MultipleSusis(n_parallel_processes=2, simulation_parameter_list=simus)
    assert multiple is not None


def test_different_experiment_ids_raise(valid_susi_params, another_valid_susi_params):
    """Different experiment_ids in the same batch should raise error."""
    simus = [
        SimulationParams(
            susi_params=valid_susi_params,
            metadata=SimulationMetaData(
                experiment_id="exp_one",
                stand_id="stand_A",
                scenario_id="scenario_1",
            ),
        ),
        SimulationParams(
            susi_params=another_valid_susi_params,
            metadata=SimulationMetaData(
                experiment_id="exp_two",
                stand_id="stand_B",
                scenario_id="scenario_1",
            ),
        ),
    ]
    with pytest.raises(ValueError, match="same experiment_id"):
        MultipleSusis(n_parallel_processes=2, simulation_parameter_list=simus)


def test_missing_stand_id_in_batch_raises():
    """Missing stand_id in batch run should raise error at metadata creation."""
    with pytest.raises(ValidationError, match="stand_id and scenario_id"):
        SimulationMetaData(
            experiment_id="batch_exp",
            scenario_id="scenario_1",
        )


def test_missing_scenario_id_in_batch_raises():
    """Missing scenario_id in batch run should raise error at metadata creation."""
    with pytest.raises(ValidationError, match="stand_id and scenario_id"):
        SimulationMetaData(
            experiment_id="batch_exp",
            stand_id="stand_A",
        )

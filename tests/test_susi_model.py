import datetime
import pytest
from pathlib import Path

from susi.io.susi_parameter_model import (
    AllometryFileAndSpecies,
    CanopyLayerAllometry,
    CanopyLayerName,
    CanopyParams,
    CuttingManagementParams,
    StandardNPKFertilizationParameters,
    NutrientFertilizationParameters,
    OrganicLayerParams,
    OutputParams,
    PeatTemperatureParams,
    PeatTypes,
    SimulationConfig,
    SiteParams,
    SusiParams,
    TreeSpecies,
    WeatherParams,
    get_photo_parameters_by_location,
    LocationsForPhotoParams,
    h_mor_from_drainage_and_mass_mor_Pitkanen,
    Thinning,
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
            allometry_file_registry={
                1: AllometryFileAndSpecies(filename="test_allometry.csv", species_id=1)
            },
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
                application_yr=2004, management_type=Thinning(target_basal_area={'dominant': 12})
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


def test_valid_canopy_layer_pointers_length(test_data_path):
    """Test that valid canopy layer pointers with correct length pass validation."""
    params = SusiParams(
        weather_parameters=WeatherParams(
            FMI_weather_filepath=test_data_path / "weather.csv",
        ),
        simulation_config=SimulationConfig(
            start_date=datetime.datetime(2004, 1, 1),
            end_date=datetime.datetime(2007, 12, 31),
        ),
        allometry_parameters=CanopyLayerAllometry(
            allometry_dir_path=test_data_path,
            allometry_file_registry={
                1: AllometryFileAndSpecies(filename="test_allometry.csv", species_id=1)
            },
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
                application_yr=2004, management_type=Thinning(target_basal_area={'dominant': 12})
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
    assert params.site_parameters.n == 5


def test_invalid_canopy_layer_pointers_length(test_data_path):
    """Test that invalid canopy layer pointers with wrong length raises error."""
    from pydantic import ValidationError

    with pytest.raises((ValueError, ValidationError), match="elements"):
        SusiParams(
            weather_parameters=WeatherParams(
                FMI_weather_filepath=test_data_path / "weather.csv",
            ),
            simulation_config=SimulationConfig(
                start_date=datetime.datetime(2004, 1, 1),
                end_date=datetime.datetime(2007, 12, 31),
            ),
            allometry_parameters=CanopyLayerAllometry(
                allometry_dir_path=test_data_path,
                allometry_file_registry={
                    1: AllometryFileAndSpecies(
                        filename="test_allometry.csv", species_id=1
                    )
                },
                pointers={
                    CanopyLayerName.dominant: [1, 1, 1],
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
                    application_yr=2004, management_type=Thinning(target_basal_area={'dominant': 12})
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


def test_valid_stand_age_all_layers(valid_susi_params):
    """Test that valid stand ages within allometry range pass validation."""
    params = valid_susi_params
    ages = params.site_parameters.initial_canopylayer_age_years
    assert ages[CanopyLayerName.dominant] == 70.0
    assert ages[CanopyLayerName.subdominant] == 70.0
    assert ages[CanopyLayerName.under] == 70.0


def test_initial_dominant_age_below_minimum(test_data_path):
    """Test that initial dominant age below minimum in allometry raises error."""
    with pytest.raises(ValueError, match="dominant.*below minimum"):
        SusiParams(
            weather_parameters=WeatherParams(
                FMI_weather_filepath=test_data_path / "weather.csv",
            ),
            simulation_config=SimulationConfig(
                start_date=datetime.datetime(2004, 1, 1),
                end_date=datetime.datetime(2007, 12, 31),
            ),
            allometry_parameters=CanopyLayerAllometry(
                allometry_dir_path=test_data_path,
                allometry_file_registry={
                    1: AllometryFileAndSpecies(
                        filename="test_allometry.csv", species_id=1
                    )
                },
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
                    CanopyLayerName.dominant: 1.0,
                    CanopyLayerName.subdominant: 20.0,
                    CanopyLayerName.under: 10.0,
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
                    application_yr=2004, management_type=Thinning(target_basal_area={'dominant': 12})
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


def test_initial_age_plus_duration_above_maximum(test_data_path):
    """Test that initial age plus simulation duration above maximum raises error."""
    with pytest.raises(ValueError, match="exceeds maximum"):
        SusiParams(
            weather_parameters=WeatherParams(
                FMI_weather_filepath=test_data_path / "weather.csv",
            ),
            simulation_config=SimulationConfig(
                start_date=datetime.datetime(2004, 1, 1),
                end_date=datetime.datetime(2100, 12, 31),
            ),
            allometry_parameters=CanopyLayerAllometry(
                allometry_dir_path=test_data_path,
                allometry_file_registry={
                    1: AllometryFileAndSpecies(
                        filename="test_allometry.csv", species_id=1
                    )
                },
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
                    CanopyLayerName.dominant: 80.0,
                    CanopyLayerName.subdominant: 20.0,
                    CanopyLayerName.under: 10.0,
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
                    application_yr=2004, management_type=Thinning(target_basal_area={'dominant': 12})
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


def test_subdominant_layer_validation(test_data_path):
    """Test that subdominant layer validation works."""
    with pytest.raises(ValueError, match="subdominant.*below minimum"):
        SusiParams(
            weather_parameters=WeatherParams(
                FMI_weather_filepath=test_data_path / "weather.csv",
            ),
            simulation_config=SimulationConfig(
                start_date=datetime.datetime(2004, 1, 1),
                end_date=datetime.datetime(2007, 12, 31),
            ),
            allometry_parameters=CanopyLayerAllometry(
                allometry_dir_path=test_data_path,
                allometry_file_registry={
                    1: AllometryFileAndSpecies(
                        filename="test_allometry.csv", species_id=1
                    )
                },
                pointers={
                    CanopyLayerName.dominant: None,
                    CanopyLayerName.subdominant: [1, 1, 1, 1, 1],
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
                    CanopyLayerName.dominant: 40.0,
                    CanopyLayerName.subdominant: 1.0,
                    CanopyLayerName.under: 10.0,
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
                    application_yr=2004, management_type=Thinning(target_basal_area={'dominant': 12})
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


def test_under_layer_validation(test_data_path):
    """Test that understorey layer validation works."""
    with pytest.raises(ValueError, match="under.*below minimum"):
        SusiParams(
            weather_parameters=WeatherParams(
                FMI_weather_filepath=test_data_path / "weather.csv",
            ),
            simulation_config=SimulationConfig(
                start_date=datetime.datetime(2004, 1, 1),
                end_date=datetime.datetime(2007, 12, 31),
            ),
            allometry_parameters=CanopyLayerAllometry(
                allometry_dir_path=test_data_path,
                allometry_file_registry={
                    1: AllometryFileAndSpecies(
                        filename="test_allometry.csv", species_id=1
                    )
                },
                pointers={
                    CanopyLayerName.dominant: None,
                    CanopyLayerName.subdominant: None,
                    CanopyLayerName.under: [1, 1, 1, 1, 1],
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
                    CanopyLayerName.dominant: 40.0,
                    CanopyLayerName.subdominant: 20.0,
                    CanopyLayerName.under: 1.0,
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
                    application_yr=2004, management_type=Thinning(target_basal_area={'dominant': 12})
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


def test_valid_allometry_pointers_correspondence(test_data_path):
    """Test that valid allometry pointers correspondence passes validation."""
    params = SusiParams(
        weather_parameters=WeatherParams(
            FMI_weather_filepath=test_data_path / "weather.csv",
        ),
        simulation_config=SimulationConfig(
            start_date=datetime.datetime(2004, 1, 1),
            end_date=datetime.datetime(2007, 12, 31),
        ),
        allometry_parameters=CanopyLayerAllometry(
            allometry_dir_path=test_data_path,
            allometry_file_registry={
                1: AllometryFileAndSpecies(filename="test_allometry.csv", species_id=1),
                2: AllometryFileAndSpecies(filename="test_allometry.csv", species_id=1),
                3: AllometryFileAndSpecies(filename="test_allometry.csv", species_id=1),
            },
            pointers={
                CanopyLayerName.dominant: [1, 1, 1, 1, 1],
                CanopyLayerName.subdominant: [2, 2, 2, 2, 2],
                CanopyLayerName.under: [3, 3, 3, 3, 3],
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
                application_yr=2004, management_type=Thinning(target_basal_area={'dominant': 12})
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
    assert params is not None


def test_invalid_allometry_pointers_missing_key(test_data_path):
    """Test that allometry pointers referencing non-existent key raises error."""
    from pydantic import ValidationError

    with pytest.raises((ValueError, ValidationError), match="not exist"):
        SusiParams(
            weather_parameters=WeatherParams(
                FMI_weather_filepath=test_data_path / "weather.csv",
            ),
            simulation_config=SimulationConfig(
                start_date=datetime.datetime(2004, 1, 1),
                end_date=datetime.datetime(2007, 12, 31),
            ),
            allometry_parameters=CanopyLayerAllometry(
                allometry_dir_path=test_data_path,
                allometry_file_registry={
                    1: AllometryFileAndSpecies(
                        filename="test_allometry.csv", species_id=1
                    )
                },
                pointers={
                    CanopyLayerName.dominant: [99, 99, 99, 99, 99],
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
                    application_yr=2004, management_type=Thinning(target_basal_area={'dominant': 12})
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

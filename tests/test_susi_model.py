import datetime
import pytest
from pathlib import Path

from susi.io.susi_parameter_model import (
    AllometryParams,
    CanopyParams,
    FertilizationParameters,
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
        allometry_parameters=AllometryParams(
            allometry_dir_path=test_data_path,
            dominant={1: "test_allometry.xlsx"},
            subdominant={0: "test_allometry.xlsx"},
            under={0: "test_allometry.xlsx"},
        ),
        canopy_parameters=CanopyParams(),
        organic_layer_parameters=OrganicLayerParams(),
        output_parameters=OutputParams(),
        photo_parameters=get_photo_parameters_by_location(
            location=LocationsForPhotoParams("All_data")
        ),
        site_parameters=SiteParams(
            L=10.0,
            initial_dominant_stand_age_years=70.0,
            initial_subdominant_stand_age_years=70.0,
            initial_understorey_age_years=70.0,
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
            fertilization=FertilizationParameters(
                application_year=2201,
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
    assert params.site_parameters.initial_dominant_stand_age_years == 70.0
    assert params.site_parameters.initial_subdominant_stand_age_years == 70.0
    assert params.site_parameters.initial_understorey_age_years == 70.0


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
            allometry_parameters=AllometryParams(
                allometry_dir_path=test_data_path,
                dominant={1: "test_allometry.xlsx"},
                subdominant={0: "test_allometry.xlsx"},
                under={0: "test_allometry.xlsx"},
            ),
            canopy_parameters=CanopyParams(),
            organic_layer_parameters=OrganicLayerParams(),
            output_parameters=OutputParams(),
            photo_parameters=get_photo_parameters_by_location(
                location=LocationsForPhotoParams("All_data")
            ),
            site_parameters=SiteParams(
                L=10.0,
                initial_dominant_stand_age_years=1.0,
                initial_subdominant_stand_age_years=20.0,
                initial_understorey_age_years=10.0,
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
                fertilization=FertilizationParameters(
                    application_year=2201,
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
            allometry_parameters=AllometryParams(
                allometry_dir_path=test_data_path,
                dominant={1: "test_allometry.xlsx"},
                subdominant={0: "test_allometry.xlsx"},
                under={0: "test_allometry.xlsx"},
            ),
            canopy_parameters=CanopyParams(),
            organic_layer_parameters=OrganicLayerParams(),
            output_parameters=OutputParams(),
            photo_parameters=get_photo_parameters_by_location(
                location=LocationsForPhotoParams("All_data")
            ),
            site_parameters=SiteParams(
                L=10.0,
                initial_dominant_stand_age_years=80.0,
                initial_subdominant_stand_age_years=20.0,
                initial_understorey_age_years=10.0,
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
                fertilization=FertilizationParameters(
                    application_year=2201,
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
            allometry_parameters=AllometryParams(
                allometry_dir_path=test_data_path,
                dominant={0: "test_allometry.xlsx"},
                subdominant={1: "test_allometry.xlsx"},
                under={0: "test_allometry.xlsx"},
            ),
            canopy_parameters=CanopyParams(),
            organic_layer_parameters=OrganicLayerParams(),
            output_parameters=OutputParams(),
            photo_parameters=get_photo_parameters_by_location(
                location=LocationsForPhotoParams("All_data")
            ),
            site_parameters=SiteParams(
                L=10.0,
                initial_dominant_stand_age_years=40.0,
                initial_subdominant_stand_age_years=1.0,
                initial_understorey_age_years=10.0,
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
                fertilization=FertilizationParameters(
                    application_year=2201,
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
            allometry_parameters=AllometryParams(
                allometry_dir_path=test_data_path,
                dominant={0: "test_allometry.xlsx"},
                subdominant={0: "test_allometry.xlsx"},
                under={1: "test_allometry.xlsx"},
            ),
            canopy_parameters=CanopyParams(),
            organic_layer_parameters=OrganicLayerParams(),
            output_parameters=OutputParams(),
            photo_parameters=get_photo_parameters_by_location(
                location=LocationsForPhotoParams("All_data")
            ),
            site_parameters=SiteParams(
                L=10.0,
                initial_dominant_stand_age_years=40.0,
                initial_subdominant_stand_age_years=20.0,
                initial_understorey_age_years=1.0,
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
                fertilization=FertilizationParameters(
                    application_year=2201,
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

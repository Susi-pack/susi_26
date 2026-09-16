import pytest
import numpy as np
import datetime
from pathlib import Path
from susi.io.susi_parameter_model import (
    AllometryFileAndSpecies,
    SusiParams,
    WeatherParams,
    SimulationConfig,
    CanopyLayerAllometry,
    CanopyLayerName,
    CanopyParams,
    OrganicLayerParams,
    OutputParams,
    SiteParams,
    NutrientFertilizationParameters,
    PeatTemperatureParams,
    TreeSpecies,
    LocationsForPhotoParams,
    get_photo_parameters_by_location,
    PeatTypes,
    StandardNPKFertilizationParameters,
    AshFertilizationParameters,
    Thinning,
    CuttingManagementParams,
)


class TestSusiMainFertilizationIntegration:
    """Test fertilization integration in Susi.main() loop."""

    @pytest.fixture
    def test_data_path(self, tmp_path):
        """Create a temporary test data path with a dummy weather file."""
        # Create a dummy weather.csv file
        weather_file = tmp_path / "weather.csv"
        weather_file.write_text("dummy,data\n")
        # Copy test_allometry.csv to tmp_path
        import shutil

        source_allometry = Path(__file__).parent.parent / "data" / "test_allometry.csv"
        shutil.copy(source_allometry, tmp_path / "test_allometry.csv")
        return tmp_path

    @pytest.fixture
    def mock_weather_data(self):
        """Create mock weather forcing data."""
        import pandas as pd

        dates = pd.date_range(start="2004-01-01", end="2007-12-31", freq="D")
        data = {
            "T": np.full(len(dates), 10.0),  # temperature
            "AT": np.full(len(dates), 10.0),  # air temperature
            "LALA": np.full(len(dates), 10.0),  # placeholder for other columns
        }
        # Create DataFrame with enough columns for indices used in susi_main.py
        columns_needed = max([4, 7, 8, 10, 13, 14]) + 1  # max index used + 1
        data = {i: np.full(len(dates), 0.0) for i in range(columns_needed)}
        data[4] = np.full(len(dates), 10.0)  # air temperature
        data[7] = np.full(len(dates), 1.0)  # precipitation
        data[8] = np.full(len(dates), 10.0)  # solar radiation
        data[10] = np.full(len(dates), 5.0)  # PAR
        data[13] = np.full(len(dates), 0.5)  # vapor pressure deficit
        data[14] = np.full(len(dates), 180)  # day of year
        return pd.DataFrame(data, index=dates)

    @pytest.fixture
    def base_susi_params(self, test_data_path: Path):
        """Create base SusiParams for testing."""
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
                peat_type=[PeatTypes.generic] * 4,
                peat_type_bottom=[PeatTypes.generic],
                anisotropy=10.0,
                vonP=True,
                vonP_top=[2, 5, 5, 6],
                vonP_bottom=7,
                bd_top=None,
                bd_bottom=0.16,
                peatN=None,
                peatP=None,
                peatK=None,
                enable_peattop=True,
                enable_peatmiddle=True,
                enable_peatbottom=True,
                rho_mor=90.0,
                h_mor=lambda drain_age, rho_mor: 0.1,  # Simple callable for testing
                cutting_management=CuttingManagementParams(
                    application_yr=2004, management_type=Thinning(target_basal_area={CanopyLayerName.dominant: 12})
                ),
                depoN=4.0,
                depoP=0.1,
                depoK=1.0,
                fertilization=None,  # Will be modified in tests
                peat_temperature=PeatTemperatureParams(),
            ),
        )

    def test_fertilization_initialization_none(self, base_susi_params, test_data_path):
        """Test that fertilization is initialized as NoFertilization when None."""
        from susi.core.fertilization import initialize_fertilization, NoFertilization

        result = initialize_fertilization(
            fertilization_params=None, n_cols=4, simulation_end_year=2005
        )
        assert isinstance(result, NoFertilization)
        assert result.ncols == 4

    def test_fertilization_initialization_standard_npk(
        self, base_susi_params, test_data_path
    ):
        """Test that fertilization is initialized as StandardNPKFertilization when provided."""
        from susi.core.fertilization import (
            initialize_fertilization,
            StandardNPKFertilization,
        )

        fert_params = StandardNPKFertilizationParameters(
            application_year=2005,
            N=NutrientFertilizationParameters(dose=50.0, decay_k=0.2, eff=0.8),
            P=NutrientFertilizationParameters(dose=30.0, decay_k=0.1, eff=0.9),
            K=NutrientFertilizationParameters(dose=40.0, decay_k=0.3, eff=0.7),
            pH_increment=0.5,
        )

        result = initialize_fertilization(
            fertilization_params=fert_params, n_cols=4, simulation_end_year=2005
        )
        assert isinstance(result, StandardNPKFertilization)
        assert result.ncols == 4
        assert result.fpara == fert_params

    def test_fertilization_initialization_ash(self, base_susi_params, test_data_path):
        """Test that fertilization is initialized as AshFertilization when provided."""
        from susi.core.fertilization import (
            initialize_fertilization,
            AshFertilization,
        )

        fert_params = AshFertilizationParameters(
            application_year=2005,
            grain_radius=0.005,
            particle_cracking_rate=2.0,
            dissolution_rate=0.005,
            K_dissolution_rate=0.00012,
            P_dissolution_rate=0.000045,
            fertilizer_dose=100.0,
            K_in_ash=50.0,
            P_in_ash=20.0,
            time_exp=1.0,
        )

        result = initialize_fertilization(
            fertilization_params=fert_params, n_cols=4, simulation_end_year=2005
        )
        assert isinstance(result, AshFertilization)
        assert result.ncols == 4
        assert result.fpara == fert_params

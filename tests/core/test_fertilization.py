import pytest
import numpy as np
from susi.core.fertilization import (
    StandardNPKFertilization,
    NoFertilization,
    AshFertilization,
    initialize_fertilization,
    FertilizationEffect,
)
from susi.io.susi_parameter_model import (
    StandardNPKFertilizationParameters,
    NutrientFertilizationParameters,
    FertilizationParameters,
    AshFertilizationParameters,
)
from susi.io.susi_parameter_model import (
    StandardNPKFertilizationParameters,
    NutrientFertilizationParameters,
    FertilizationParameters,
)


class TestStandardNPKFertilization:
    """Test StandardNPKFertilization mathematical correctness."""

    def test_compute_ph_effect(self):
        """Test pH effect calculation matches original formula."""
        # Arrange
        params = StandardNPKFertilizationParameters(
            application_year=2005,
            N=NutrientFertilizationParameters(dose=0.0, decay_k=0.5, eff=1.0),
            P=NutrientFertilizationParameters(dose=0.0, decay_k=0.5, eff=1.0),
            K=NutrientFertilizationParameters(dose=0.0, decay_k=0.5, eff=1.0),
            pH_increment=2.0,
        )
        fert = StandardNPKFertilization(n_cols=3, fpara=params)

        # Act & Assert
        # Year of application (t=0)
        assert fert.compute_ph_effect(0) == 2.0  # 2.0 * exp(0) = 2.0

        # One year after application (t=1)
        expected = 2.0 * np.exp(-0.1 * 1)
        assert fert.compute_ph_effect(1) == expected

        # Five years after application (t=5)
        expected = 2.0 * np.exp(-0.1 * 5)
        assert fert.compute_ph_effect(5) == expected

        # Ten years after application (t=10)
        expected = 2.0 * np.exp(-0.1 * 10)
        assert fert.compute_ph_effect(10) == expected

    def test_compute_nutrient_release(self):
        """Test nutrient release calculation matches original formula."""
        # Arrange
        dose = 100.0
        decay_k = 0.3
        eff = 1.0
        n_cols = 4

        params = StandardNPKFertilizationParameters(
            application_year=2005,
            N=NutrientFertilizationParameters(dose=dose, decay_k=decay_k, eff=eff),
            P=NutrientFertilizationParameters(dose=dose, decay_k=decay_k, eff=eff),
            K=NutrientFertilizationParameters(dose=dose, decay_k=decay_k, eff=eff),
            pH_increment=0.0,
        )
        fert = StandardNPKFertilization(n_cols=n_cols, fpara=params)

        # Act & Assert
        # Test t=0 (year of application)
        # release = (dose * exp(0) - dose * exp(-decay_k * 1)) * eff * ones(n_cols)
        expected_release = (dose * 1.0 - dose * np.exp(-decay_k * 1)) * eff
        expected_array = np.full(n_cols, expected_release)

        release_dict = fert.compute_nutrient_release(0)
        np.testing.assert_array_almost_equal(release_dict["N"], expected_array)
        np.testing.assert_array_almost_equal(release_dict["P"], expected_array)
        np.testing.assert_array_almost_equal(release_dict["K"], expected_array)

        # Test t=1
        # release = (dose * exp(-decay_k * 1) - dose * exp(-decay_k * 2)) * eff * ones(n_cols)
        expected_release = (
            dose * np.exp(-decay_k * 1) - dose * np.exp(-decay_k * 2)
        ) * eff
        expected_array = np.full(n_cols, expected_release)

        release_dict = fert.compute_nutrient_release(1)
        np.testing.assert_array_almost_equal(release_dict["N"], expected_array)
        np.testing.assert_array_almost_equal(release_dict["P"], expected_array)
        np.testing.assert_array_almost_equal(release_dict["K"], expected_array)

        # Test t=5
        expected_release = (
            dose * np.exp(-decay_k * 5) - dose * np.exp(-decay_k * 6)
        ) * eff
        expected_array = np.full(n_cols, expected_release)

        release_dict = fert.compute_nutrient_release(5)
        np.testing.assert_array_almost_equal(release_dict["N"], expected_array)
        np.testing.assert_array_almost_equal(release_dict["P"], expected_array)
        np.testing.assert_array_almost_equal(release_dict["K"], expected_array)

    def test_compute_nutrient_release_negative_years(self):
        """Test that negative years since fertilization triggers assertion."""
        # Arrange
        params = StandardNPKFertilizationParameters(
            application_year=2005,
            N=NutrientFertilizationParameters(dose=10.0, decay_k=0.5, eff=1.0),
            P=NutrientFertilizationParameters(dose=10.0, decay_k=0.5, eff=1.0),
            K=NutrientFertilizationParameters(dose=10.0, decay_k=0.5, eff=1.0),
            pH_increment=0.0,
        )
        fert = StandardNPKFertilization(n_cols=1, fpara=params)

        # Act & Assert
        with pytest.raises(AssertionError):
            fert.compute_nutrient_release(
                -1
            )  # Should trigger assert years_since_fertilization >= 0

    def test_compute_effect_active(self):
        """Test compute_effect returns correct values when active."""
        # Arrange
        params = StandardNPKFertilizationParameters(
            application_year=2005,
            N=NutrientFertilizationParameters(dose=100.0, decay_k=0.3, eff=1.0),
            P=NutrientFertilizationParameters(dose=100.0, decay_k=0.3, eff=1.0),
            K=NutrientFertilizationParameters(dose=100.0, decay_k=0.3, eff=1.0),
            pH_increment=1.5,
        )
        fert = StandardNPKFertilization(n_cols=2, fpara=params)

        # Act
        effect = fert.compute_effect(year=2005)  # t=0

        # Assert
        assert effect.is_active == True
        assert effect.pH_increment == 1.5  # pH_increment * exp(-0.1 * 0) = 1.5
        assert isinstance(effect.nutrient_release, dict)
        assert "N" in effect.nutrient_release
        assert "P" in effect.nutrient_release
        assert "K" in effect.nutrient_release
        assert effect.nutrient_release["N"].shape == (2,)
        assert effect.nutrient_release["P"].shape == (2,)
        assert effect.nutrient_release["K"].shape == (2,)

    def test_compute_effect_inactive(self):
        """Test compute_effect returns inactive effect when not active."""
        # Arrange
        params = StandardNPKFertilizationParameters(
            application_year=2005,
            N=NutrientFertilizationParameters(dose=100.0, decay_k=0.3, eff=1.0),
            P=NutrientFertilizationParameters(dose=100.0, decay_k=0.3, eff=1.0),
            K=NutrientFertilizationParameters(dose=100.0, decay_k=0.3, eff=1.0),
            pH_increment=1.5,
        )
        fert = StandardNPKFertilization(n_cols=2, fpara=params)

        # Act
        effect = fert.compute_effect(year=2004)  # Before application year

        # Assert
        assert effect.is_active == False
        assert effect.pH_increment == 0.0
        assert isinstance(effect.nutrient_release, dict)
        assert effect.nutrient_release["N"].shape == (2,)
        assert effect.nutrient_release["P"].shape == (2,)
        assert effect.nutrient_release["K"].shape == (2,)
        assert np.all(effect.nutrient_release["N"] == 0.0)
        assert np.all(effect.nutrient_release["P"] == 0.0)
        assert np.all(effect.nutrient_release["K"] == 0.0)


class TestNoFertilization:
    """Test NoFertilization class."""

    def test_compute_effect_always_inactive(self):
        """Test that NoFertilization always returns inactive effect."""
        # Arrange
        fert = NoFertilization(n_cols=3, fpara=None)

        # Act
        effect = fert.compute_effect(year=2020)  # Any year

        # Assert
        assert effect.is_active == False
        assert effect.pH_increment == 0.0
        assert isinstance(effect.nutrient_release, dict)
        assert effect.nutrient_release["N"].shape == (3,)
        assert effect.nutrient_release["P"].shape == (3,)
        assert effect.nutrient_release["K"].shape == (3,)
        assert np.all(effect.nutrient_release["N"] == 0.0)
        assert np.all(effect.nutrient_release["P"] == 0.0)
        assert np.all(effect.nutrient_release["K"] == 0.0)

    def test_abstract_methods_implemented(self):
        """Test that NoFertilization implements the abstract methods."""
        # Arrange
        fert = NoFertilization(n_cols=3, fpara=None)

        # Act & Assert
        # These should not raise NotImplementedError
        assert fert.compute_ph_effect(5) == 0.0
        nutrient_release = fert.compute_nutrient_release(5)
        assert isinstance(nutrient_release, dict)
        assert "N" in nutrient_release
        assert "P" in nutrient_release
        assert "K" in nutrient_release


class TestInitializeFertilization:
    """Test initialize_fertilization factory function."""

    def test_returns_no_fertilization_when_none(self):
        """Test that None fertilization parameters returns NoFertilization."""
        # Arrange
        fert_params = None
        n_cols = 5

        # Act
        fert = initialize_fertilization(fert_params, n_cols)

        # Assert
        assert isinstance(fert, NoFertilization)
        assert fert.ncols == n_cols
        assert fert.fpara is None

    def test_returns_standard_npk_fertilization(self):
        """Test that StandardNPK parameters returns StandardNPKFertilization."""
        # Arrange
        fert_params = StandardNPKFertilizationParameters(
            application_year=2005,
            N=NutrientFertilizationParameters(dose=10.0, decay_k=0.5, eff=1.0),
            P=NutrientFertilizationParameters(dose=10.0, decay_k=0.5, eff=1.0),
            K=NutrientFertilizationParameters(dose=10.0, decay_k=0.5, eff=1.0),
            pH_increment=1.0,
        )
        n_cols = 4

        # Act
        fert = initialize_fertilization(fert_params, n_cols)

        # Assert
        assert isinstance(fert, StandardNPKFertilization)
        assert fert.ncols == n_cols
        assert fert.fpara == fert_params

    def test_returns_ash_fertilization(self):
        """Test that Ash parameters returns AshFertilization."""
        # Arrange
        from susi.io.susi_parameter_model import AshFertilizationParameters

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
        n_cols = 3

        # Act
        fert = initialize_fertilization(fert_params, n_cols)

        # Assert
        assert isinstance(
            fert, AshFertilization
        )  # Will raise NotImplementedError when used
        assert fert.ncols == n_cols
        assert fert.fpara == fert_params

import pytest
import numpy as np
from susi.core.fertilization import (
    StandardNPKFertilization,
    NoFertilization,
    AshFertilization,
    initialize_fertilization,
)
from susi.io.susi_parameter_model import (
    StandardNPKFertilizationParameters,
    NutrientFertilizationParameters,
)


class TestStandardNPKFertilization:
    """Test StandardNPKFertilization mathematical correctness."""

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

        # Application year
        effect = fert.compute_effect(year=2005)

        # Assert
        assert effect.is_active
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
        assert not effect.is_active
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
        fert = NoFertilization(n_cols=3)

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
        fert = NoFertilization(n_cols=3)

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

import numpy as np
import pytest
from susi.core.fertilization import (
    StandardNPKFertilization,
    AshFertilization,
    initialize_fertilization,
)
from susi.io.susi_parameter_model import (
    StandardNPKFertilizationParameters,
    NutrientFertilizationParameters,
    AshFertilizationParameters,
    FertilizationParameters,
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


class TestInitializeFertilization:
    """Test initialize_fertilization factory function."""

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
        fert = initialize_fertilization(fert_params, n_cols, simulation_end_year=0)

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
        fert = initialize_fertilization(fert_params, n_cols, simulation_end_year=2006)

        # Assert
        assert isinstance(
            fert, AshFertilization
        )  # Will raise NotImplementedError when used
        assert fert.ncols == n_cols
        assert fert.fpara == fert_params


class TestAshFertilization:
    """Test AshFertilization over the simulated years."""

    def test_compute_effect_in_last_simulation_year(self):
        """
        The simulation loop runs up to and including the end year
        (see susi_main), so the effect must be computable in that year.
        """
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
        fert = initialize_fertilization(fert_params, n_cols=3, simulation_end_year=2006)

        # Act
        effect = fert.compute_effect(year=2006)

        # Assert
        assert effect.is_active


# ---------------------------------------------------------------------------
# Pinned values
#
# Characterization tests: they record, bit for bit, the values the current
# implementation produces, so that a refactor can be shown to change nothing.
# The expected numbers were generated by running `effect_as_floats` and pasting
# the `repr` of each float, so they round-trip exactly. They are compared with
# `==`, never `pytest.approx`.
#
# `effect_as_floats` is the only place that knows the fertilization API. When
# the API changes, only the helper changes, never a pinned number.
# ---------------------------------------------------------------------------


def effect_as_floats(
    params: FertilizationParameters, year: int
) -> tuple[float, float, float, float]:
    """(pH_increment, N, P, K) for one year, as plain floats."""
    fertilization = initialize_fertilization(
        params, n_cols=1, simulation_end_year=2030
    )
    assert fertilization is not None  # only None params give None
    effect = fertilization.compute_effect(year)
    return (
        float(effect.pH_increment),
        float(effect.nutrient_release["N"][0]),
        float(effect.nutrient_release["P"][0]),
        float(effect.nutrient_release["K"][0]),
    )


# The closed form, with all three nutrients different.
FIRST_ORDER_DECAY = StandardNPKFertilizationParameters(
    application_year=2005,
    pH_increment=1.5,
    N=NutrientFertilizationParameters(dose=100.0, decay_k=0.3, eff=0.8),
    P=NutrientFertilizationParameters(dose=40.0, decay_k=0.2, eff=1.0),
    K=NutrientFertilizationParameters(dose=60.0, decay_k=0.5, eff=0.9),
)

# The K storage runs out in 2013: `min(dK, K_storage)` trims that year's release
# to 0.1473, so 2014 is the first year with zero K. P is also fully released
# by the end of 2013.
ASH_K_RUNS_OUT = AshFertilizationParameters(
    application_year=2005,
    fertilizer_dose=3000.0,
    K_in_ash=5.0,
    P_in_ash=2.0,
    time_exp=2.0,
)

# Same as above but dissolving 20x faster: the ash dissolves completely during
# 2015, so `min(dm, mass)` caps the pH increment at its maximum,
# 3000 * 2.5 / 15000 = 0.5, from 2016 on.
ASH_FULLY_DISSOLVED = AshFertilizationParameters(
    application_year=2005,
    fertilizer_dose=3000.0,
    K_in_ash=5.0,
    P_in_ash=2.0,
    time_exp=2.0,
    dissolution_rate=0.1,
)

# year: (pH_increment, N, P, K)
FIRST_ORDER_DECAY_PINNED = {
    2004: (0.0, 0.0, 0.0, 0.0),
    2005: (1.5, 20.73454234546257, 7.2507698768807245, 21.247344375517798),
    2006: (1.3572561270539394, 15.360526767015319, 5.936428281693701, 12.887165801224318),
    2007: (1.2280961296169728, 11.379358108274182, 4.860336397664518, 7.816461175242675),
    2013: (0.6739934461758323, 1.8809952439730182, 1.4639051909227536, 0.38915868692656114),
    2014: (0.6098544896108986, 1.3934755497508668, 1.1985441993989534, 0.23603667511446935),
    2015: (0.5518191617571635, 1.0323120773299146, 0.9812849949711531, 0.14316348027355563),
    2016: (0.49930662554711935, 0.7647555963157956, 0.8034082029168563, 0.08683304013707623),
    2030: (0.1231274979358482, 0.011467951293443746, 0.04885530313298783, 7.918148331527132e-05),
}

ASH_K_RUNS_OUT_PINNED = {
    2004: (0.0, 0.0, 0.0, 0.0),
    2005: (0.0, 0.0, 0.08100000000000006, 0.21600000000000014),
    2006: (0.0015, 0.0, 0.11658845378625908, 0.3109025434300242),
    2007: (0.0036590454404862765, 0.0, 0.16766378271987156, 0.44710342058632413),
    2013: (0.03369962628144382, 0.0, 0.18022018080203306, 0.14725381547208816),
    2014: (0.04093428977043944, 0.0, 0.0, 0.0),
    2015: (0.048674576114955015, 0.0, 0.0, 0.0),
    2016: (0.056881299734230804, 0.0, 0.0, 0.0),
    2030: (0.20244046690025613, 0.0, 0.0, 0.0),
}

ASH_FULLY_DISSOLVED_PINNED = {
    2004: (0.0, 0.0, 0.0, 0.0),
    2005: (0.0, 0.0, 0.08100000000000006, 0.21600000000000014),
    2006: (0.03, 0.0, 0.11210130615999996, 0.29893681642666653),
    2007: (0.07151900228148149, 0.0, 0.15201049160391228, 0.40536131094376604),
    2013: (0.4416554272177852, 0.0, 0.09773164951411525, 0.26061773203764066),
    2014: (0.47785233444523534, 0.0, 0.055392740923719944, 0.14771397579658652),
    2015: (0.4983681644169835, 0.0, 0.010439698773460042, 0.027839196729226776),
    2016: (0.5, 0.0, 0.0, 0.0),
    2030: (0.5, 0.0, 0.0, 0.0),
}


@pytest.mark.parametrize("year, expected", FIRST_ORDER_DECAY_PINNED.items())
def test_pinned_first_order_decay(year: int, expected: tuple[float, ...]):
    assert effect_as_floats(FIRST_ORDER_DECAY, year) == expected


@pytest.mark.parametrize("year, expected", ASH_K_RUNS_OUT_PINNED.items())
def test_pinned_ash_K_runs_out(year: int, expected: tuple[float, ...]):
    assert effect_as_floats(ASH_K_RUNS_OUT, year) == expected


@pytest.mark.parametrize("year, expected", ASH_FULLY_DISSOLVED_PINNED.items())
def test_pinned_ash_fully_dissolved(year: int, expected: tuple[float, ...]):
    assert effect_as_floats(ASH_FULLY_DISSOLVED, year) == expected

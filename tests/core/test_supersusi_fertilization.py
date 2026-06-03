import math

import numpy as np
import pytest
from typing import cast

from supersusi.core import fertilization as fertilization_dispatcher
from supersusi.core.fertilization_models import (
    ash as ash_module,
    no_fertilization,
    npk as npk_module,
)
from supersusi.core.fertilization_types import Inputs, Nutrient, State
from susi.io.susi_parameter_model import (
    AshFertilizationParameters,
    NutrientFertilizationParameters,
    StandardNPKFertilizationParameters,
)

ApplicationYear = int


def _make_npk_params(
    application_year: ApplicationYear = 2005,
    n_dose: float = 100.0,
    p_dose: float = 100.0,
    k_dose: float = 100.0,
    n_decay: float = 0.3,
    p_decay: float = 0.3,
    k_decay: float = 0.3,
    n_eff: float = 1.0,
    p_eff: float = 1.0,
    k_eff: float = 1.0,
    ph_increment: float = 1.5,
) -> StandardNPKFertilizationParameters:
    return StandardNPKFertilizationParameters(
        application_year=application_year,
        N=NutrientFertilizationParameters(dose=n_dose, decay_k=n_decay, eff=n_eff),
        P=NutrientFertilizationParameters(dose=p_dose, decay_k=p_decay, eff=p_eff),
        K=NutrientFertilizationParameters(dose=k_dose, decay_k=k_decay, eff=k_eff),
        pH_increment=ph_increment,
    )


def _make_ash_params(
    application_year: ApplicationYear = 2005,
    grain_radius: float = 0.005,
    particle_cracking_rate: float = 2.0,
    dissolution_rate: float = 0.005,
    k_dissolution_rate: float = 0.00012,
    p_dissolution_rate: float = 0.000045,
    density: float = 1000.0,
    fertilizer_dose: float = 100.0,
    k_in_ash: float = 50.0,
    p_in_ash: float = 20.0,
    time_exp: float = 1.0,
) -> AshFertilizationParameters:
    return AshFertilizationParameters(
        application_year=application_year,
        grain_radius=grain_radius,
        particle_cracking_rate=particle_cracking_rate,
        dissolution_rate=dissolution_rate,
        K_dissolution_rate=k_dissolution_rate,
        P_dissolution_rate=p_dissolution_rate,
        density=density,
        fertilizer_dose=fertilizer_dose,
        K_in_ash=k_in_ash,
        P_in_ash=p_in_ash,
        time_exp=time_exp,
    )


class TestFertilizationDispatcher:
    """Tests for the dispatcher in `supersusi.core.fertilization`."""

    def test_compute_static_inputs_npk_returns_none(self) -> None:
        """NPK has no static inputs."""
        params = npk_module.Params(n_cols=2, fpara=_make_npk_params())
        assert fertilization_dispatcher.compute_static_inputs(params) is None

    def test_compute_static_inputs_no_fertilization_returns_none(self) -> None:
        """No-fertilization has no static inputs."""
        params = no_fertilization.Params(n_cols=2)
        assert fertilization_dispatcher.compute_static_inputs(params) is None

    def test_compute_static_inputs_ash_returns_static_inputs(self) -> None:
        """Ash precomputes a `StaticInputs` instance of the right shape."""
        params = ash_module.Params(
            n_cols=2,
            simulation_end_year=2010,
            fpara=_make_ash_params(application_year=2005),
        )
        static = fertilization_dispatcher.compute_static_inputs(params)
        assert static is not None
        assert isinstance(static, ash_module.ComputedConstants)
        assert static.pH_history.shape == (5,)
        assert static.K_release_history.shape == (5,)
        assert static.P_release_history.shape == (5,)

    @pytest.mark.parametrize(
        "calendar_year,expected",
        [
            (2004, -1),
            (2005, 0),
            (2006, 1),
            (2015, 10),
        ],
    )
    def test_compute_dynamic_inputs_ash_offsets_year(
        self, calendar_year: int, expected: int
    ) -> None:
        """For ash, `years_since_fertilization = calendar_year - application_year`."""
        params = ash_module.Params(
            n_cols=2,
            simulation_end_year=2010,
            fpara=_make_ash_params(application_year=2005),
        )
        dynamic = fertilization_dispatcher.compute_dynamic_inputs(
            params, calendar_year=calendar_year
        )
        assert dynamic.years_since_fertilization == expected

    @pytest.mark.parametrize(
        "calendar_year,expected",
        [
            (2004, -1),
            (2005, 0),
            (2006, 1),
        ],
    )
    def test_compute_dynamic_inputs_npk_offsets_year(
        self, calendar_year: int, expected: int
    ) -> None:
        """For NPK, `years_since_fertilization = calendar_year - application_year`."""
        params = npk_module.Params(n_cols=2, fpara=_make_npk_params())
        dynamic = fertilization_dispatcher.compute_dynamic_inputs(
            params, calendar_year=calendar_year
        )
        assert dynamic.years_since_fertilization == expected

    def test_compute_dynamic_inputs_no_fertilization_returns_zero(self) -> None:
        """For no-fertilization, the offset is irrelevant; we just need a value."""
        params = no_fertilization.Params(n_cols=2)
        dynamic = fertilization_dispatcher.compute_dynamic_inputs(
            params, calendar_year=2020
        )
        assert dynamic.years_since_fertilization == 0

    def test_run_timestep_dispatches_to_npk(self) -> None:
        """Dispatcher delegates to `npk.run_timestep` for NPK params."""
        params = npk_module.Params(n_cols=2, fpara=_make_npk_params())
        dynamic = Inputs(years_since_fertilization=0)
        state = fertilization_dispatcher.run_timestep(
            params=params, computed_constants=None, inputs=dynamic
        )
        assert state.pH_increment == pytest.approx(1.5)

    def test_run_timestep_dispatches_to_no_fertilization(self) -> None:
        """Dispatcher delegates to `no_fertilization.run_timestep`."""
        params = no_fertilization.Params(n_cols=2)
        dynamic = Inputs(years_since_fertilization=999)
        state = fertilization_dispatcher.run_timestep(
            params=params, computed_constants=None, inputs=dynamic
        )
        assert state.pH_increment == 0.0
        assert np.all(state.nutrient_release["N"] == 0.0)
        assert np.all(state.nutrient_release["P"] == 0.0)
        assert np.all(state.nutrient_release["K"] == 0.0)

    def test_run_timestep_dispatches_to_ash(self) -> None:
        """Dispatcher delegates to `ash.run_timestep` for ash params."""
        params = ash_module.Params(
            n_cols=2,
            simulation_end_year=2010,
            fpara=_make_ash_params(application_year=2005),
        )
        static = ash_module.compute_constants(params)
        dynamic = Inputs(years_since_fertilization=0)
        state = fertilization_dispatcher.run_timestep(
            params=params, computed_constants=static, inputs=dynamic
        )
        assert isinstance(state, State)
        assert np.all(state.nutrient_release["N"] == 0.0)


class TestNPK:
    """Tests for `supersusi.core.fertilization_models.npk`."""

    @pytest.mark.parametrize("n_cols", [1, 2, 7])
    def test_inactive_year_returns_unfertilized_state(self, n_cols: int) -> None:
        """For `years_since_fertilization < 0`, pH is 0 and all releases are 0."""
        params = npk_module.Params(n_cols=n_cols, fpara=_make_npk_params())
        state = npk_module.run_timestep(
            params=params,
            inputs=Inputs(years_since_fertilization=-1),
        )
        assert state.pH_increment == 0.0
        for nutrient in ("N", "P", "K"):
            assert state.nutrient_release[nutrient].shape == (n_cols,)
            assert np.all(state.nutrient_release[nutrient] == 0.0)

    def test_application_year_pH_is_full(self) -> None:
        """At `t = 0` the pH effect is the full `pH_increment` parameter."""
        params = npk_module.Params(n_cols=1, fpara=_make_npk_params())
        state = npk_module.run_timestep(
            params=params,
            inputs=Inputs(years_since_fertilization=0),
        )
        assert state.pH_increment == pytest.approx(1.5)

    def test_pH_decays_exponentially(self) -> None:
        """For `t > 0`, `pH_increment = pH_increment_param * exp(-0.1 * t)`."""
        params = npk_module.Params(n_cols=1, fpara=_make_npk_params())
        for t, expected in [
            (1, 1.5 * math.exp(-0.1)),
            (3, 1.5 * math.exp(-0.3)),
            (10, 1.5 * math.exp(-1.0)),
        ]:
            state = npk_module.run_timestep(
                params=params,
                inputs=Inputs(years_since_fertilization=t),
            )
            assert state.pH_increment == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("n_cols", [1, 2, 7])
    def test_application_year_nutrient_release_formula(self, n_cols: int) -> None:
        """At `t = 0`, `release = dose * (1 - exp(-k)) * eff` per nutrient."""
        params = npk_module.Params(
            n_cols=n_cols,
            fpara=_make_npk_params(n_dose=100.0, p_dose=200.0, k_dose=300.0),
        )
        state = npk_module.run_timestep(
            params=params,
            inputs=Inputs(years_since_fertilization=0),
        )
        for nutrient, dose in [("N", 100.0), ("P", 200.0), ("K", 300.0)]:
            expected = dose * (1.0 - math.exp(-0.3)) * 1.0
            assert state.nutrient_release[cast(Nutrient, nutrient)] == pytest.approx(
                np.full(n_cols, expected), rel=1e-12
            )

    def test_nutrient_release_decay_at_later_year(self) -> None:
        """At `t = 1`, `release = dose * (exp(-k) - exp(-2k)) * eff` per nutrient."""
        params = npk_module.Params(
            n_cols=3,
            fpara=_make_npk_params(n_dose=10.0, p_dose=20.0, k_dose=40.0),
        )
        state = npk_module.run_timestep(
            params=params,
            inputs=Inputs(years_since_fertilization=1),
        )
        for nutrient, dose in [("N", 10.0), ("P", 20.0), ("K", 40.0)]:
            expected = dose * (math.exp(-0.3) - math.exp(-0.6))
            assert state.nutrient_release[cast(Nutrient, nutrient)] == pytest.approx(
                np.full(3, expected), rel=1e-12
            )

    def test_per_nutrient_decay_and_eff_independent(self) -> None:
        """Each nutrient uses its own `decay_k` and `eff` independently."""
        params = npk_module.Params(
            n_cols=2,
            fpara=_make_npk_params(
                n_dose=10.0,
                p_dose=20.0,
                k_dose=40.0,
                n_decay=0.1,
                p_decay=0.5,
                k_decay=1.0,
                n_eff=0.5,
                p_eff=0.8,
                k_eff=1.0,
            ),
        )
        state = npk_module.run_timestep(
            params=params,
            inputs=Inputs(years_since_fertilization=2),
        )
        expected_n = 10.0 * (math.exp(-0.2) - math.exp(-0.3)) * 0.5
        expected_p = 20.0 * (math.exp(-1.0) - math.exp(-1.5)) * 0.8
        expected_k = 40.0 * (math.exp(-2.0) - math.exp(-3.0)) * 1.0
        assert state.nutrient_release["N"] == pytest.approx(
            np.full(2, expected_n), rel=1e-12
        )
        assert state.nutrient_release["P"] == pytest.approx(
            np.full(2, expected_p), rel=1e-12
        )
        assert state.nutrient_release["K"] == pytest.approx(
            np.full(2, expected_k), rel=1e-12
        )

    def test_zero_pH_increment_is_honored(self) -> None:
        """`pH_increment = 0.0` (the field default) yields no pH effect."""
        params = npk_module.Params(n_cols=1, fpara=_make_npk_params(ph_increment=0.0))
        state = npk_module.run_timestep(
            params=params,
            inputs=Inputs(years_since_fertilization=0),
        )
        assert state.pH_increment == 0.0
        # Nutrient release is unaffected by pH_increment.
        assert state.nutrient_release["N"][0] == pytest.approx(
            100.0 * (1.0 - math.exp(-0.3)), rel=1e-12
        )


class TestAshPrecompute:
    """Tests for `ash.compute_static_inputs` (the precompute loop)."""

    def test_output_shape_matches_simulation_window(self) -> None:
        """Output arrays have length `simulation_end_year - application_year`."""
        params = ash_module.Params(
            n_cols=2,
            simulation_end_year=2010,
            fpara=_make_ash_params(application_year=2005),
        )
        static = ash_module.compute_constants(params)
        assert static.pH_history.shape == (5,)
        assert static.K_release_history.shape == (5,)
        assert static.P_release_history.shape == (5,)

    def test_first_year_pH_is_zero(self) -> None:
        """At `t = 0` no ash has dissolved, so `pH_increment = 0`."""
        params = ash_module.Params(
            n_cols=2,
            simulation_end_year=2010,
            fpara=_make_ash_params(application_year=2005),
        )
        static = ash_module.compute_constants(params)
        assert static.pH_history[0] == 0.0

    def test_first_year_release_values(self) -> None:
        """Hand-computed values for a typical SUSI parameter set.

        At `t = 0` no ash has dissolved yet, so `pH_history[0] = 0`.
        The K and P release values at `t = 0` are the per-year fluxes
        (non-zero) and equal the precompute formula applied to the grain
        geometry:
            A_total = n_particles_0 * 4 * pi * r^2 = 60 m^2
            dK = 0.00012 * 60 = 0.0072
            dP = 0.000045 * 60 = 0.0027
        """
        params = ash_module.Params(
            n_cols=2,
            simulation_end_year=2010,
            fpara=_make_ash_params(
                application_year=2005,
                grain_radius=0.005,
                density=1000.0,
                fertilizer_dose=100.0,
                k_in_ash=50.0,
                p_in_ash=20.0,
                particle_cracking_rate=2.0,
                time_exp=1.0,
                dissolution_rate=0.005,
                k_dissolution_rate=0.00012,
                p_dissolution_rate=0.000045,
            ),
        )
        static = ash_module.compute_constants(params)
        assert static.K_release_history[0] == pytest.approx(0.0072, rel=1e-12)
        assert static.P_release_history[0] == pytest.approx(0.0027, rel=1e-12)
        # pH_history[0] is always 0 because dissolved_ash is 0 at t=0.
        assert static.pH_history[0] == 0.0
        # At t=1: dissolved_ash = 0.7333 -> pH increment = 5.0000e-05.
        assert static.pH_history[1] == pytest.approx(5.0e-05, rel=1e-12)
        # At t=1, both K and P releases should be larger (more particles -> more area).
        assert static.K_release_history[1] > static.K_release_history[0]
        assert static.P_release_history[1] > static.P_release_history[0]

    def test_release_arrays_are_float64(self) -> None:
        """Result arrays are float64 (regression test for the int64 truncation bug)."""
        params = ash_module.Params(
            n_cols=2,
            simulation_end_year=2010,
            fpara=_make_ash_params(application_year=2005),
        )
        static = ash_module.compute_constants(params)
        assert static.pH_history.dtype == np.float64
        assert static.K_release_history.dtype == np.float64
        assert static.P_release_history.dtype == np.float64

    def test_pH_history_is_monotonically_non_decreasing(self) -> None:
        """Dissolved ash is cumulative, so `pH_history` never decreases."""
        params = ash_module.Params(
            n_cols=2,
            simulation_end_year=2015,
            fpara=_make_ash_params(application_year=2005),
        )
        static = ash_module.compute_constants(params)
        diffs = np.diff(static.pH_history)
        assert np.all(diffs >= -1e-15)

    def test_cumulative_K_release_capped_by_K_in_ash(self) -> None:
        """The sum of all `K_release` should never exceed `K_in_ash`."""
        k_in_ash = 50.0
        params = ash_module.Params(
            n_cols=2,
            simulation_end_year=2015,
            fpara=_make_ash_params(
                application_year=2005,
                fertilizer_dose=100.0,
                k_in_ash=k_in_ash,
            ),
        )
        static = ash_module.compute_constants(params)
        assert static.K_release_history.sum() <= k_in_ash + 1e-9

    def test_cumulative_P_release_capped_by_P_in_ash(self) -> None:
        """The sum of all `P_release` should never exceed `P_in_ash`."""
        p_in_ash = 20.0
        params = ash_module.Params(
            n_cols=2,
            simulation_end_year=2015,
            fpara=_make_ash_params(
                application_year=2005,
                fertilizer_dose=100.0,
                p_in_ash=p_in_ash,
            ),
        )
        static = ash_module.compute_constants(params)
        assert static.P_release_history.sum() <= p_in_ash + 1e-9

    def test_simulation_end_year_equal_application_year(self) -> None:
        """Edge case: zero-length simulation window produces empty arrays."""
        params = ash_module.Params(
            n_cols=2,
            simulation_end_year=2005,
            fpara=_make_ash_params(application_year=2005),
        )
        static = ash_module.compute_constants(params)
        assert static.pH_history.shape == (0,)
        assert static.K_release_history.shape == (0,)
        assert static.P_release_history.shape == (0,)


class TestAshRunTimestep:
    """Tests for `ash.run_timestep` (the lookup step)."""

    def _setup(
        self, end_year: int = 2010
    ) -> tuple[ash_module.Params, ash_module.ComputedConstants]:
        params = ash_module.Params(
            n_cols=3,
            simulation_end_year=end_year,
            fpara=_make_ash_params(application_year=2005),
        )
        static = ash_module.compute_constants(params)
        return params, static

    @pytest.mark.parametrize("t", [-5, -1, -100])
    def test_inactive_year_returns_unfertilized_state(self, t: int) -> None:
        """For `t < 0`, pH is 0 and nutrient releases are zeros of the right shape."""
        params, static = self._setup()
        state = ash_module.run_timestep(
            params=params,
            computed_constants=static,
            inputs=Inputs(years_since_fertilization=t),
        )
        assert state.pH_increment == 0.0
        for nutrient in ("N", "P", "K"):
            assert state.nutrient_release[nutrient].shape == (3,)
            assert np.all(state.nutrient_release[nutrient] == 0.0)

    def test_active_year_nutrient_shapes(self) -> None:
        """At `t >= 0`, N is zeros, P and K are scalars broadcast to `n_cols`."""
        params, static = self._setup()
        state = ash_module.run_timestep(
            params=params,
            computed_constants=static,
            inputs=Inputs(years_since_fertilization=0),
        )
        assert state.nutrient_release["N"].shape == (3,)
        assert np.all(state.nutrient_release["N"] == 0.0)
        assert state.nutrient_release["P"].shape == (3,)
        assert state.nutrient_release["K"].shape == (3,)

    def test_active_year_pH_matches_precompute(self) -> None:
        """`pH_increment` at year `t` equals the precomputed `pH_history[t]`."""
        params, static = self._setup()
        for t in (0, 1, 2, 4):
            state = ash_module.run_timestep(
                params=params,
                computed_constants=static,
                inputs=Inputs(years_since_fertilization=t),
            )
            assert state.pH_increment == pytest.approx(static.pH_history[t], rel=1e-12)

    def test_active_year_K_release_matches_precompute(self) -> None:
        """`nutrient_release["K"]` equals `K_release_history[t] * ones`."""
        params, static = self._setup()
        for t in (0, 1, 3):
            state = ash_module.run_timestep(
                params=params,
                computed_constants=static,
                inputs=Inputs(years_since_fertilization=t),
            )
            assert state.nutrient_release["K"] == pytest.approx(
                np.full(3, static.K_release_history[t]), rel=1e-12
            )

    def test_active_year_P_release_matches_precompute(self) -> None:
        """`nutrient_release["P"]` equals `P_release_history[t] * ones`."""
        params, static = self._setup()
        for t in (0, 1, 3):
            state = ash_module.run_timestep(
                params=params,
                computed_constants=static,
                inputs=Inputs(years_since_fertilization=t),
            )
            assert state.nutrient_release["P"] == pytest.approx(
                np.full(3, static.P_release_history[t]), rel=1e-12
            )


class TestNoFertilization:
    """Tests for `supersusi.core.fertilization_models.no_fertilization`."""

    @pytest.mark.parametrize("n_cols", [1, 2, 5])
    @pytest.mark.parametrize("t", [-1, 0, 100])
    def test_run_timestep_always_returns_zeros(self, n_cols: int, t: int) -> None:
        """No fertilization: always `pH_increment=0` and zeros of the right shape."""
        params = no_fertilization.Params(n_cols=n_cols)
        state = no_fertilization.run_timestep(
            params=params, inputs=Inputs(years_since_fertilization=t)
        )
        assert state.pH_increment == 0.0
        for nutrient in ("N", "P", "K"):
            assert state.nutrient_release[nutrient].shape == (n_cols,)
            assert np.all(state.nutrient_release[nutrient] == 0.0)

    @pytest.mark.parametrize("n_cols", [1, 2, 5])
    def test_unfertilized_state_helper(self, n_cols: int) -> None:
        """`unfertilized_state` returns the same zero-filled `State`."""
        state = no_fertilization.unfertilized_state(n_cols=n_cols)
        assert state.pH_increment == 0.0
        for nutrient in ("N", "P", "K"):
            assert state.nutrient_release[nutrient].shape == (n_cols,)
            assert np.all(state.nutrient_release[nutrient] == 0.0)

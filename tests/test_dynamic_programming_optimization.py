import numpy as np
import pytest

from analysis.optimization.dynamic_programming import (
    get_minimum_values_per_variable,
    shift_points_to_positive_values,
    undo_shift_to_positive_values,
    PartialParetoPoint,
    recover_scenario_choices,
)


def assert_arrays_equal(original, restored):
    assert len(original) == len(restored)
    for orig, rest in zip(original, restored):
        np.testing.assert_allclose(orig, rest, rtol=1e-5, atol=1e-8)


# --- Fixtures ---


@pytest.fixture
def original_data():
    return [
        np.array([[-642.92734375, 9517.09333333], [-646.17617188, 12541.13333333]]),
        np.array([[-83.98195553, 26035.66388889], [-141.52484245, 36018.57972222]]),
        np.array([[-1626.45367432, 18815.07913194], [-1689.78857727, 29029.22809028]]),
        np.array(
            [
                [-1991.18133545, 50868.14583333],
                [-2370.94451904, 65349.98611111],
                [-3229.86724854, 71108.22222222],
                [-3262.39379883, 89875.22222222],
            ]
        ),
        np.array([[-347.32382202, 15556.54722222], [-369.33560181, 18892.81666667]]),
    ]


@pytest.fixture
def simple_data():
    return [
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        np.array([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]),
    ]


@pytest.fixture
def negative_data():
    return [
        np.array([[-1000.0, -500.0], [-200.0, -100.0], [-50.0, -10.0]]),
        np.array([[1000.0, 500.0]]),
    ]


@pytest.fixture
def single_array_data():
    return [
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    ]


@pytest.fixture
def large_variable_data():
    rng = np.random.default_rng(42)
    return [rng.uniform(-1e4, 1e4, size=(rng.integers(1, 10), 2)) for _ in range(20)]


# --- Tests ---


@pytest.mark.parametrize(
    "fixture_name",
    [
        "original_data",
        "simple_data",
        "negative_data",
        "single_array_data",
        "large_variable_data",
    ],
)
def test_roundtrip_identity(fixture_name, request):
    arrays = request.getfixturevalue(fixture_name)
    min_vals = get_minimum_values_per_variable(arrays)
    shifted = shift_points_to_positive_values(arrays, min_vals)
    restored = undo_shift_to_positive_values(shifted, min_vals)
    assert_arrays_equal(arrays, restored)


def test_structure_preserved(original_data):
    min_vals = get_minimum_values_per_variable(original_data)
    shifted = shift_points_to_positive_values(original_data, min_vals)
    restored = undo_shift_to_positive_values(shifted, min_vals)
    assert [arr.shape for arr in restored] == [arr.shape for arr in original_data]


def test_single_point_array():
    arrays = [np.array([[5.0, 10.0]]), np.array([[3.0, 7.0], [8.0, 2.0]])]
    min_vals = get_minimum_values_per_variable(arrays)
    shifted = shift_points_to_positive_values(arrays, min_vals)
    restored = undo_shift_to_positive_values(shifted, min_vals)
    assert_arrays_equal(arrays, restored)


def test_recover_scenario_choices():
    # Single point (no parent)
    p0 = PartialParetoPoint(
        objective_vector=(1.0,),
        parent_point=None,
        current_scenario_choice=3,
        stand_index=0,
    )
    assert recover_scenario_choices(p0) == [3]

    # Linear chain: p0 -> p1 -> p2
    p1 = PartialParetoPoint(
        objective_vector=(2.0,),
        parent_point=p0,
        current_scenario_choice=7,
        stand_index=1,
    )
    p2 = PartialParetoPoint(
        objective_vector=(3.0,),
        parent_point=p1,
        current_scenario_choice=1,
        stand_index=2,
    )
    assert recover_scenario_choices(p2) == [3, 7, 1]

    # Intermediate node returns only its own ancestry
    assert recover_scenario_choices(p1) == [3, 7]

    # Chain with repeated choice values
    p3 = PartialParetoPoint(
        objective_vector=(4.0,),
        parent_point=p2,
        current_scenario_choice=3,  # same as p0
        stand_index=3,
    )
    assert recover_scenario_choices(p3) == [3, 7, 1, 3]

    # Choice of zero is valid
    p_zero = PartialParetoPoint(
        objective_vector=(0.0,),
        parent_point=None,
        current_scenario_choice=0,
        stand_index=0,
    )
    assert recover_scenario_choices(p_zero) == [0]

    print("All tests passed.")


test_recover_scenario_choices()

import numpy as np
import pytest

from analysis.optimization.dynamic_programming import (
    get_minimum_values_per_variable,
    shift_points_to_positive_values,
    undo_shift_to_positive_values,
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

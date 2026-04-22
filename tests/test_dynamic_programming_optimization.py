import numpy as np
import pytest

from analysis.optimization.dynamic_programming import (
    PartialParetoPoint,
    recover_scenario_choices,
    get_minimum_values_per_variable,
    shift_points_to_positive_values,
)


def test_get_minimum_values_per_variable():
    data = [
        np.array([[1, 2, 3], [2, 3, 4]]),
        np.array([[0, 2, 3], [2, 3, 4], [-1, 3, 0]]),
    ]
    assert get_minimum_values_per_variable(data) == (-1, 2, 0)


def test_shift_points_to_positive_values():
    data = [
        np.array([[1, 2, 3], [2, 3, 4]]),
        np.array([[0, 2, 3], [2, 3, 4], [-1, 3, 0]]),
    ]
    for arr in shift_points_to_positive_values(
        data, minimum_values_per_variable=get_minimum_values_per_variable(data)
    ):
        assert np.all(arr) > 0


def test_recover_scenario_choices():
    # Single point (no parent)
    p0 = PartialParetoPoint(
        objective_vector=(1.0,),
        parent_point=None,
        current_scenario_choice=3,
        stand_index=0,
    )
    assert recover_scenario_choices(p0) == (3,)

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
    assert recover_scenario_choices(p2) == (3, 7, 1)

    # Intermediate node returns only its own ancestry
    assert recover_scenario_choices(p1) == (3, 7)

    # Chain with repeated choice values
    p3 = PartialParetoPoint(
        objective_vector=(4.0,),
        parent_point=p2,
        current_scenario_choice=3,  # same as p0
        stand_index=3,
    )
    assert recover_scenario_choices(p3) == (3, 7, 1, 3)

    # Choice of zero is valid
    p_zero = PartialParetoPoint(
        objective_vector=(0.0,),
        parent_point=None,
        current_scenario_choice=0,
        stand_index=0,
    )
    assert recover_scenario_choices(p_zero) == (0,)

    print("All tests passed.")


test_recover_scenario_choices()

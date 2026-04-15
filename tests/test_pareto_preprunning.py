import numpy as np

from analysis.optimization import prepruning


def test_basic_dominance():
    A = np.array(
        [
            [1, 2],
            [2, 3],
            [1, 3],
            [3, 1],
        ]
    )

    res = prepruning.preprune_pareto_dominated_scenarios(A)

    expected = [0, 3]
    assert res == expected


def test_no_dominance():
    A = np.array(
        [
            [1, 3],
            [2, 2],
            [3, 1],
        ]
    )

    res = prepruning.preprune_pareto_dominated_scenarios(A)

    expected = [0, 1, 2]
    assert res == expected


def test_chain_dominance():
    A = np.array(
        [
            [5, 5],
            [4, 4],
            [3, 3],
            [2, 2],
        ]
    )

    res = prepruning.preprune_pareto_dominated_scenarios(A)

    expected = [3]
    assert res == expected


def test_duplicates_treated_as_different():
    """
    Duplicated scenarios should remain.
    """
    A = np.array(
        [
            [1, 2],
            [1, 2],
            [2, 3],
        ]
    )

    res = prepruning.preprune_pareto_dominated_scenarios(A)

    expected = [0, 1]
    assert res == expected


def test_high_dimensional_case():
    A = np.array(
        [
            [1, 2, 3, 4],
            [1, 2, 3, 5],
            [2, 2, 3, 4],
            [0, 5, 5, 5],
        ]
    )

    res = prepruning.preprune_pareto_dominated_scenarios(A)

    expected = [0, 3]

    assert res == expected


def test_single_point():
    A = np.array([[1, 2, 3]])
    res = prepruning.preprune_pareto_dominated_scenarios(A)
    expected = [0]
    assert res == expected


def test_empty_scenarios():
    A = np.array([]).reshape(0, 3)
    assert prepruning.preprune_pareto_dominated_scenarios(A) == []


def test_single_optimization_variable():
    A = np.array([[3], [1], [2]])
    assert prepruning.preprune_pareto_dominated_scenarios(A) == [1]


def test_negative_values():
    A = np.array([[-1, -5], [-2, -3], [0, 0]])
    assert prepruning.preprune_pareto_dominated_scenarios(A) == [0, 1]


def test_eps_tolerance():
    A = np.array([[1.0], [1.0 + 1e-10]])
    assert prepruning.preprune_pareto_dominated_scenarios(A) == [0, 1]


def test_two_scenarios():
    A = np.array([[1, 2], [2, 1]])
    assert prepruning.preprune_pareto_dominated_scenarios(A) == [0, 1]


def test_partial_ties():
    A = np.array([[1, 2], [2, 1], [2, 3]])
    assert prepruning.preprune_pareto_dominated_scenarios(A) == [0, 1]

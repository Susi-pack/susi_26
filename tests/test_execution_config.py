import pytest

from susi.io.execution_config import MultipleSusis, SimulationParams
from susi.io.metadata_model import SimulationMetaData
from inputs.parameters import golden_test


@pytest.fixture
def two_duplicate_simus() -> list[SimulationParams]:
    return [
        SimulationParams(
            susi_params=golden_test.PARAMETERS, metadata=SimulationMetaData()
        )
        for _ in range(2)
    ]


@pytest.fixture
def two_different_simus() -> list[SimulationParams]:
    return [
        SimulationParams(
            susi_params=golden_test.PARAMETERS, metadata=SimulationMetaData()
        ),
        # Create two different simulations
        SimulationParams(
            susi_params=golden_test.PARAMETERS,
            metadata=SimulationMetaData(parameter_output_filename="lalala.json"),
        ),
    ]


@pytest.fixture
def one_hundred_different_simus() -> list[SimulationParams]:
    return [
        SimulationParams(
            susi_params=golden_test.PARAMETERS,
            metadata=SimulationMetaData(parameter_output_filename=str(i) + ".json"),
        )
        for i in range(100)
    ]


def test_duplicate_models(two_duplicate_simus):
    """
    Computing the same twice would not make sense
    Make sure there are no duplicated simulation parameters
    """
    with pytest.raises(ValueError):
        MultipleSusis(
            n_parallel_processes=1,
            simulation_parameter_list=two_duplicate_simus,
        )


def test_maximum_number_of_parallel_processes_validation(one_hundred_different_simus):
    # No more than 40 cores are allowed in multiprocessing
    with pytest.raises(ValueError):
        MultipleSusis(
            n_parallel_processes=41,
            simulation_parameter_list=one_hundred_different_simus,
        )


def test_less_parallel_processes_than_simus(two_different_simus):
    with pytest.raises(ValueError):
        MultipleSusis(
            n_parallel_processes=3,
            simulation_parameter_list=two_different_simus,
        )

import pytest

from susi.io.execution_config import ExecutionConfig, SimulationParams
from susi.io.metadata_model import SimulationMetaData
from inputs.parameters import golden_test


def test_duplicate_models():
    N_RUNS = 2

    duplicate_simus = [
        SimulationParams(
            susi_params=golden_test.PARAMETERS, metadata=SimulationMetaData()
        )
        for _ in range(N_RUNS)
    ]

    with pytest.raises(ValueError):
        ExecutionConfig(
            n_runs=N_RUNS,
            n_parallel_processes=1,
            simulation_parameter_list=duplicate_simus,
        )


def test_different_number_of_models():
    simus = [
        SimulationParams(
            susi_params=golden_test.PARAMETERS, metadata=SimulationMetaData()
        ),
        SimulationParams(
            susi_params=golden_test.PARAMETERS,
            metadata=SimulationMetaData(parameter_output_filename="lalala.json"),
        ),
    ]

    with pytest.raises(ValueError):
        ExecutionConfig(
            n_runs=1,
            n_parallel_processes=1,
            simulation_parameter_list=simus,
        )

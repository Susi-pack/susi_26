import pytest

from susi.io.execution_config import MultipleSusis, SimulationParams
from susi.io.metadata_model import SimulationMetaData
from inputs.parameters import golden_test
from tests import meaningless_susi_model


@pytest.fixture
def two_duplicate_susi_params() -> list[SimulationParams]:
    return [
        SimulationParams(
            susi_params=golden_test.PARAMETERS,
            metadata=SimulationMetaData(experiment_id=str(i)),
        )
        for i in range(2)
    ]


@pytest.fixture
def two_duplicate_experiment_folder_paths() -> list[SimulationParams]:
    return [
        SimulationParams(
            susi_params=golden_test.PARAMETERS,
            metadata=SimulationMetaData(experiment_id=str("THE_SAME")),
        ),
        SimulationParams(
            susi_params=meaningless_susi_model.PARAMETERS,
            metadata=SimulationMetaData(experiment_id=str("THE_SAME")),
        ),
    ]


@pytest.fixture
def two_valid_simus() -> list[SimulationParams]:
    return [
        SimulationParams(
            susi_params=golden_test.PARAMETERS,
            metadata=SimulationMetaData(experiment_id="one"),
        ),
        # Create two different Susi params and output folders
        SimulationParams(
            susi_params=meaningless_susi_model.PARAMETERS,
            metadata=SimulationMetaData(experiment_id="two"),
        ),
    ]


@pytest.fixture
def one_hundred_valid_simus() -> list[SimulationParams]:
    # Make sure we create different Susi parameters
    # and different experiment_ids so that the other errors
    # do not shade this one
    return [
        SimulationParams(
            susi_params=golden_test.PARAMETERS.model_copy(
                update={"params_schema_version": i}
            ),
            metadata=SimulationMetaData(experiment_id=str(i)),
        )
        for i in range(100)
    ]


def test_duplicate_susi_params(two_duplicate_susi_params):
    """
    Computing the same twice would not make sense
    Make sure there are no duplicated simulation parameters
    """
    with pytest.raises(ValueError):
        MultipleSusis(
            n_parallel_processes=1,
            simulation_parameter_list=two_duplicate_susi_params,
        )


def test_duplicate_folder_names(two_duplicate_experiment_folder_paths):
    """
    Storing Susi results twice in the same folder
    would rewrite the previous contents of the folder
    """
    with pytest.raises(ValueError):
        MultipleSusis(
            n_parallel_processes=1,
            simulation_parameter_list=two_duplicate_experiment_folder_paths,
        )


def test_maximum_number_of_parallel_processes_validation(one_hundred_valid_simus):
    # No more than 40 cores are allowed in multiprocessing
    with pytest.raises(ValueError):
        MultipleSusis(
            n_parallel_processes=41,
            simulation_parameter_list=one_hundred_valid_simus,
        )


def test_less_parallel_processes_than_simus(two_valid_simus):
    with pytest.raises(ValueError):
        MultipleSusis(
            n_parallel_processes=3,
            simulation_parameter_list=two_valid_simus,
        )

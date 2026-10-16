"""
Tests for the optimization's handling of a target spec's direction.

Go through `core.run_optimization` over a tiny run folder written to disk:
one target variable, one stand with two scenarios to choose between, and a
second stand with a single scenario (the Pareto search refuses a lone stand).
With a single target, the Pareto front collapses to the one best combination,
so the total it holds says which way the optimization went.
"""

import json
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from analysis.optimization import core as opti_core
from susi.io.load_output_data import (
    NetcdfVariableArray,
    NetcdfVariablePath,
    StandID,
)


def _write_scenario_folder(
    run_dirpath: Path, stand_id: str, scenario_id: str, volume: float
) -> None:
    """
    Write one scenario folder the output loader can read, whose single
    netcdf variable `/volume` holds the same value everywhere.
    """
    scenario_dirpath = run_dirpath / stand_id / scenario_id
    scenario_dirpath.mkdir(parents=True)

    netcdf_filepath = scenario_dirpath / "output.nc"
    with netCDF4.Dataset(netcdf_filepath, "w") as nc:
        nc.createDimension("time", 2)
        nc.createDimension("column", 3)
        variable = nc.createVariable("volume", "f8", ("time", "column"))
        variable.units = "m3/ha"
        variable[:] = np.full((2, 3), volume)

    (scenario_dirpath / "metadata.json").write_text(
        json.dumps(
            {
                "stand_id": stand_id,
                "scenario_id": scenario_id,
                "netcdf_output_filepath": str(netcdf_filepath),
                "timestamp_start": "2020-01-01T00:00:00",
                "timestamp_end": "2020-12-31T00:00:00",
            }
        )
    )
    (scenario_dirpath / "params.json").write_text(json.dumps({}))


@pytest.mark.parametrize(
    ("direction", "expected_front"),
    [
        # Stand 1 (2 ha) offers 1 or 3 m3/ha, i.e. 2 or 6 m3; stand 2 (1 ha)
        # always adds 5 m3. So the project total is 7 or 11 m3.
        (opti_core.Direction.MINIMIZE, [[7.0]]),
        (opti_core.Direction.MAXIMIZE, [[11.0]]),
    ],
)
def test_the_pareto_front_follows_the_direction_and_keeps_the_values_sign(
    tmp_path, direction, expected_front
):
    run_dirpath = tmp_path / "run_a"
    _write_scenario_folder(run_dirpath, "1", "small", volume=1.0)
    _write_scenario_folder(run_dirpath, "1", "large", volume=3.0)
    _write_scenario_folder(run_dirpath, "2", "only", volume=5.0)

    results = opti_core.run_optimization(
        target_specs={
            NetcdfVariablePath("/volume"): opti_core.TargetSpec(
                aggregation=NetcdfVariableArray.mean_of_all_values,
                direction=direction,
            )
        },
        run_dirpath=run_dirpath,
        stand_areas={StandID("1"): 2.0, StandID("2"): 1.0},
        epsilon=1e-7,
        n_random_points=5,
    )

    np.testing.assert_allclose(results.pareto_front.target_vectors, expected_front)

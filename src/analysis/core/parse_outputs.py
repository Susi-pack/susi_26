from typing import NewType, Any
from pathlib import Path
from collections import defaultdict

from susi.io.load_output_data import (
    StandID,
    ScenarioID,
    SimulationParamsFromJSON,
    read_params_from_jsons,
    list_subdirectories,
)


ParamName = NewType("ParamName", str)

# Scenario name is how scenarios are told apart in the first place, so it
# trivially "differs" across every scenario by definition. It is never
# useful to surface as a differing parameter, so `find_differing_params`
# unconditionally excludes it (see root CONTEXT.md, "Differing Parameters").
_SCENARIO_NAME_PARAM = "site_parameters/scenario_name"


def retrieve_scenarios_for_stand(
    stand_id: StandID, outputs_dir: Path
) -> dict[ScenarioID, Path]:
    """
    Finds all scenarios for a given stand based on folder structure.
    Scenarios are subdirectories under output_dir/stand_id/.
    """
    stand_dir = outputs_dir / stand_id
    if not stand_dir.exists() or not stand_dir.is_dir():
        return {}

    scenario_dirs = list_subdirectories(stand_dir)
    return {ScenarioID(d.name): d for d in scenario_dirs}


def retrieve_parameters_for_stand(
    stand_id: StandID, output_dir: Path
) -> dict[ScenarioID, SimulationParamsFromJSON]:
    """
    Gets susi parameters from params.json files for all scenarios of a given stand.
    """
    parameters_per_stand: dict[ScenarioID, SimulationParamsFromJSON] = {}
    scenarios = retrieve_scenarios_for_stand(stand_id, output_dir)

    for scenario_id, scenario_folderpath in scenarios.items():
        parameters_per_stand[scenario_id] = read_params_from_jsons(scenario_folderpath)

    return parameters_per_stand


def _traverse_susi_params(params: dict, parent_path: str = "") -> dict[str, Any]:
    """
    Traverse nested susi_params dict, returning flat dict with '/' separated paths.
    """
    flat = {}
    for key, value in params.items():
        current_path = f"{parent_path}/{key}" if parent_path else key
        if isinstance(value, dict):
            flat.update(_traverse_susi_params(value, current_path))
        else:
            flat[current_path] = value
    return flat


def _get_flat_params_per_scenario(
    stand_id: StandID, output_dir: Path
) -> dict[ScenarioID, dict[str, Any]]:
    """
    Returns flat parameter mappings for each scenario of a stand.
    Each scenario maps to {flat_param_path: value}.
    """
    scenario_params = retrieve_parameters_for_stand(stand_id, output_dir)
    if not scenario_params:
        return {}

    scenario_flat_params: dict[ScenarioID, dict[str, Any]] = {}
    for scen_id, params in scenario_params.items():
        scenario_flat_params[scen_id] = _traverse_susi_params(params.susi_params)

    return scenario_flat_params


def _collect_all_param_paths(flat_params: dict[ScenarioID, dict[str, Any]]) -> set[str]:
    """
    Returns set of all unique parameter paths across all scenarios.
    """
    all_param_paths = set()
    for params in flat_params.values():
        all_param_paths.update(params.keys())
    return all_param_paths


def _build_param_value_mapping(
    flat_params: dict[ScenarioID, dict[str, Any]], all_param_paths: set[str]
) -> dict[str, dict[Any, list[ScenarioID]]]:
    """
    Builds complete mapping: {param_path: {value: [ScenarioIDs with that value]}}.
    Handles unhashable types by converting lists to tuples and dicts to sorted tuples.
    """
    result: dict[str, dict[Any, list[ScenarioID]]] = {}

    for param_path in all_param_paths:
        value_to_scenarios: dict[Any, list[ScenarioID]] = defaultdict(list)
        for scen_id, params in flat_params.items():
            value = params.get(param_path)
            # Convert unhashable types to hashable equivalents
            if isinstance(value, list):
                value = tuple(value)
            elif isinstance(value, dict):
                value = tuple(sorted(value.items()))
            value_to_scenarios[value].append(scen_id)

        # Sort scenario IDs for deterministic output
        result[param_path] = {
            k: sorted(v, key=lambda x: x) for k, v in value_to_scenarios.items()
        }

    return result


def find_differing_params(
    stand_id: StandID, output_dir: Path
) -> dict[ParamName, dict[Any, list[ScenarioID]]]:
    """
    Detect parameters that differ between scenarios for the same stand.
    Returns dict mapping parameter names to dict of {param_value: [list of ScenarioIDs with that value]}.
    Only includes parameters that have differing values across scenarios.
    """
    flat_params = _get_flat_params_per_scenario(stand_id, output_dir)
    if not flat_params:
        return {}

    all_param_paths = _collect_all_param_paths(flat_params)
    value_mapping = _build_param_value_mapping(flat_params, all_param_paths)

    # Only include parameters with more than one unique value, excluding the
    # scenario name itself (it trivially always differs).
    return {
        ParamName(param_path): scen_ids
        for param_path, scen_ids in value_mapping.items()
        if len(scen_ids) > 1 and param_path != _SCENARIO_NAME_PARAM
    }


def find_unique_params(stand_id: StandID, output_dir: Path) -> dict[ParamName, Any]:
    """
    Detect parameters that have identical values across all scenarios for a stand.
    Returns dict mapping parameter names to their unique value.
    Only includes parameters that have the same value in all scenarios.
    """
    flat_params = _get_flat_params_per_scenario(stand_id, output_dir)
    if not flat_params:
        return {}

    all_param_paths = _collect_all_param_paths(flat_params)
    value_mapping = _build_param_value_mapping(flat_params, all_param_paths)

    # Only include parameters with exactly one unique value
    result: dict[ParamName, Any] = {}
    for param_path, scen_ids in value_mapping.items():
        if len(scen_ids) == 1:
            unique_value = next(iter(scen_ids.keys()))
            result[ParamName(param_path)] = unique_value
    return result

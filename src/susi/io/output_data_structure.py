from typing import NewType
from dataclasses import dataclass
import numpy as np

StandID = NewType("StandID", str)  # same as stand folder name
ScenarioID = NewType("ScenarioID", str)  # same as scenario folder name
NetcdfVariablePath = NewType(
    "NetcdfVariablePath", str
)  # Example: "/balance/K/fertilization_release"


@dataclass(frozen=True)
class NetcdfVariableInfo:
    """
    Contains all information of a variable from the Susi netcdf file,
    except its values
    """

    name: str  # Example: "fertilization_release"
    dimension_names: tuple[str]
    shape: tuple[int]
    units: str  # Unit description


@dataclass
class OutputDataStore:
    """
    The main data structure.
    SoA (struct of arrays), keyed by variable name.
    """

    stands: list[StandID]  # ["stand_A", "stand_B", ...]
    scenarios: dict[StandID, list[ScenarioID]]  # {"stand_A": ["scen_1", "scen_2"], ...}
    variables: list[NetcdfVariablePath]

    # Core data structure: Struct of Arrays, keyed by variable
    # Example: {var1: {(standA, scenA): [...]} }
    data: dict[NetcdfVariablePath, dict[tuple[StandID, ScenarioID], np.ndarray]]

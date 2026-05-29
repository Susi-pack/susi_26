from dataclasses import dataclass, field
from jaxtyping import Float, Array


@dataclass(frozen=True)
class HydrologyState:
    dwts: Float[Array, "scenarios days n_cols"] = field(doc="Water table depths, [m]")
    afps: Float[Array, "scenarios days n_cols"] = field(
        doc="Air-filled porosity, [m3 m-3]"
    )
    deltas: Float[Array, "scenarios days n_cols"] = field(doc="")
    hts: Float[Array, "scenarios days n_cols"] = field(doc="Water table depths, [m]")
    runoffwest: Float[Array, "scenarios days"] = field(
        doc="Daily runoff from west ditch, [m]"
    )
    runoffeast: Float[Array, "scenarios days"] = field(
        doc="Daily runoff from east ditch, [m]"
    )
    surfacerunoff: Float[Array, "scenarios days n_cols"] = field(
        doc="Daily surface runoff from each column, [m]"
    )
    runoff: Float[Array, "scenarios days"] = field(
        doc="Daily total runoff, sum of west, east and surface runoff, [m]"
    )

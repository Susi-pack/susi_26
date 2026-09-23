"""The shared TreeStratum value type: one species' aggregated growth-model
inputs for one stand, used identically by xml_to_allometry.py and
metsakeskus_to_allometry.py.

Implemented as a frozen pydantic dataclass rather than a stdlib dataclass or
a pydantic BaseModel:
- frozen + dataclass keeps it an immutable, hashable value object usable as
  a field inside plain (stdlib) frozen dataclasses, e.g. metsakeskus_to_
  allometry.py's PerSpecies[TreeStratum]/StandCandidate/ParsedStand.
- pydantic validates on construction, so a malformed XML value (e.g. a
  non-numeric age) is still rejected the way xml_to_allometry.py's own
  BaseModel-based TreeStratum used to reject it. See issue #278 / #277 for
  the full discussion.
"""

from dataclasses import dataclass as stdlib_dataclass
from typing import Generic, TypeVar

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class TreeStratum:
    age: float
    basal_area: float
    stem_count: float
    mean_diameter: float
    mean_height: float


# The shared "nothing recorded here" value. Both tools use this one instance
# directly wherever a species/stratum slot has no data, instead of each
# maintaining (and defensively copying) their own equivalent -- TreeStratum
# is immutable, so there is nothing a shared instance risks by being reused.
ZERO_STRATUM = TreeStratum(
    age=0,
    basal_area=0.0,
    stem_count=0,
    mean_diameter=0.0,
    mean_height=0.0,
)


T = TypeVar("T")


@stdlib_dataclass(frozen=True)
class PerSpecies(Generic[T]):
    """One value per SUSI growth-model species slot.

    Promoted here from metsakeskus_to_allometry.py, the tool that
    originated it: all three allometry-generating tools store "one
    TreeStratum per species slot" this way now, instead of
    xml_to_allometry.py/new_growth_allometry.py's previous
    tuple[TreeStratum, TreeStratum, TreeStratum] convention (index
    0/1/2, documented only in comments).

    Field stays `deciduous`, not `birch` -- this is not a naming
    inconsistency to clean up. src/susi/core (the simulation engine)
    always calls the third species-code bucket `birch` (TreeSpecies enum,
    susi_utils.py/gvegetation.py's "1 pine, 2 spruce, 3 birch"
    convention). But tools/ -- StandData, PerSpecies, TreeStratum
    aggregation -- is the raw inventory layer, where that bucket is a
    genuine mix of species, not yet collapsed into SUSI's
    birch-simplification. See CONTEXT.md's Species entry for the full
    rationale.
    """

    pine: T
    spruce: T
    deciduous: T

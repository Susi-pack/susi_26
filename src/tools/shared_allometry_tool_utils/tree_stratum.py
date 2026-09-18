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

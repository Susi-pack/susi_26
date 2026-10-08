"""Derives SUSI's peat_type ('A' generic/carex/woody vs 'S' sphagnum) from
Metsakeskus soiltype (+ fertilityclass fallback for generic peat, code 60,
where no subtype is recorded).

Shared by metsakeskus_to_allometry.py and
metsakeskus_to_allometry_stratified.py -- ported from
MK_to_susi_allometry_v2.ipynb's _SOILTYPE_TO_PEAT_TYPE / generic-peat
fallback (cell 2), which is Metsakeskus-specific and therefore does not
belong in xml_to_allometry.py or new_growth_allometry.py.
"""

from susi.io.susi_parameter_model import PeatTypes

# Sphagnum-dominated soiltype codes -> "S" (sphagnum). Everything else with a
# known subtype -- carex (61/64/66), wood-dominant (63) -- -> "A" (generic).
# Code 60 (generic peat, no subtype recorded) is NOT in this table: it falls
# through to the fertility-class-based fallback below.
_SOILTYPE_TO_PEAT_TYPE: dict[int, PeatTypes] = {
    61: PeatTypes.generic,
    62: PeatTypes.sphagnum,
    63: PeatTypes.generic,
    64: PeatTypes.generic,
    65: PeatTypes.sphagnum,
    66: PeatTypes.generic,
    67: PeatTypes.sphagnum,
}

# soiltype 60 = generic peat, no sub-type recorded. Notebook fallback:
# FC<=4 -> "A" (generic), FC>=5 -> "S" (sphagnum).
_GENERIC_PEAT_SOILTYPE = 60
_GENERIC_PEAT_FC_THRESHOLD = 4


def peat_type_from_soiltype(
    soiltype: int | None, fertilityclass: int
) -> PeatTypes | None:
    """Returns None when soiltype is missing or an unrecognized code -- the
    caller decides whether that's a skip (stratified sampling can't place a
    stand with no peat_type into a stratum) or left unset (single-stand
    processing, where peat_type is optional metadata)."""
    if soiltype is None:
        return None
    if soiltype == _GENERIC_PEAT_SOILTYPE:
        return (
            PeatTypes.generic
            if fertilityclass <= _GENERIC_PEAT_FC_THRESHOLD
            else PeatTypes.sphagnum
        )
    return _SOILTYPE_TO_PEAT_TYPE.get(soiltype)
# SusiParams checks that every allometry file a run will read exists
# (ticket 23, docs/adr/0005). It's the only place that checks:
# AllometryFileAndSpecies.file_path stays a plain Path, so building a
# CanopyLayerAllometry/StandParams/StandData, or loading stand_data.json,
# never needs the CSVs.
#
# Missing-file variants are built by re-validating a valid SusiParams with
# one field swapped: SusiParams(**{**dict(valid), field: new}) reruns every
# SusiParams model validator, while the nested models passed in as instances
# are kept as they are.
from pathlib import Path

import pydantic
import pytest

from susi.io.susi_parameter_model import (
    AllometryFileAndSpecies,
    CanopyLayerAllometry,
    CanopyLayerName,
    ClearCut,
    CuttingManagementParams,
    StandParams,
    SusiParams,
)
from tests.test_cutting_management import (
    DATA_DIR,
    _make_regeneration_allometry,
    _make_susi_params,
)

N = 5


def _stand_params_reading(*file_paths: Path) -> StandParams:
    """One registry entry per file, the first on the dominant layer and the
    second (if any) on the subdominant one."""
    registry = {}
    pointers: dict[CanopyLayerName, list[int] | None] = {
        layer: None for layer in CanopyLayerName
    }
    layers = [CanopyLayerName.dominant, CanopyLayerName.subdominant]
    for registry_number, (layer, file_path) in enumerate(
        zip(layers, file_paths, strict=False), start=1
    ):
        registry[registry_number] = AllometryFileAndSpecies(
            file_path=file_path, species_id=1
        )
        pointers[layer] = [registry_number] * N
    return StandParams(
        site_fertility_class=4,
        canopy_layer_allometry=CanopyLayerAllometry(
            allometry_file_registry=registry, pointers=pointers
        ),
    )


def _clearcut_reading(file_path: Path) -> CuttingManagementParams:
    # model_construct skips ClearCut's own validators: the age-starts-at-one
    # one reads the CSV itself, so a missing new-growth file would otherwise
    # fail inside ClearCut before SusiParams ever saw it. The wrapping
    # CuttingManagementParams is constructed too, because validating its
    # union-typed management_type re-validates the ClearCut.
    clearcut = ClearCut.model_construct(
        new_growth_allometry=CanopyLayerAllometry(
            allometry_file_registry={
                1: AllometryFileAndSpecies(file_path=file_path, species_id=1)
            },
            pointers={
                CanopyLayerName.dominant: [1] * N,
                CanopyLayerName.subdominant: None,
                CanopyLayerName.under: None,
            },
        ),
        strips_to_cut=[True] * N,
    )
    return CuttingManagementParams.model_construct(
        application_yr=2005, management_type=clearcut
    )


def _with(valid: SusiParams, **fields) -> SusiParams:
    return SusiParams(**{**dict(valid), **fields})


def _with_cutting_management(
    valid: SusiParams, cutting_management: CuttingManagementParams
) -> SusiParams:
    site_parameters = valid.site_parameters.model_copy(
        update={"cutting_management": cutting_management}
    )
    return _with(valid, site_parameters=site_parameters)


def test_all_allometry_files_present_passes():
    clearcut = CuttingManagementParams(
        application_yr=2005,
        management_type=ClearCut(
            new_growth_allometry=_make_regeneration_allometry(n=N),
            strips_to_cut=[True] * N,
        ),
    )

    params = _make_susi_params(cutting_management=clearcut, n=N)

    assert params.site_parameters.cutting_management is not None


def test_a_missing_stand_allometry_file_raises_naming_it(tmp_path):
    valid = _make_susi_params(cutting_management=None, n=N)
    missing = tmp_path / "missing_stand.csv"

    with pytest.raises(pydantic.ValidationError) as error:
        _with(valid, stand_params=_stand_params_reading(missing))

    assert str(missing) in str(error.value)


def test_a_missing_clearcut_new_growth_file_raises_naming_it(tmp_path):
    valid = _make_susi_params(cutting_management=None, n=N)
    missing = tmp_path / "missing_new_growth.csv"

    with pytest.raises(pydantic.ValidationError) as error:
        _with_cutting_management(valid, _clearcut_reading(missing))

    assert str(missing) in str(error.value)


def test_the_error_lists_every_missing_file_not_just_the_first(tmp_path):
    valid = _make_susi_params(cutting_management=None, n=N)
    missing_dominant = tmp_path / "missing_dominant.csv"
    missing_subdominant = tmp_path / "missing_subdominant.csv"
    missing_new_growth = tmp_path / "missing_new_growth.csv"
    site_parameters = valid.site_parameters.model_copy(
        update={"cutting_management": _clearcut_reading(missing_new_growth)}
    )

    with pytest.raises(pydantic.ValidationError) as error:
        _with(
            valid,
            stand_params=_stand_params_reading(missing_dominant, missing_subdominant),
            site_parameters=site_parameters,
        )

    for missing in (missing_dominant, missing_subdominant, missing_new_growth):
        assert str(missing) in str(error.value)


def test_an_existing_file_among_missing_ones_is_not_reported(tmp_path):
    valid = _make_susi_params(cutting_management=None, n=N)
    present = DATA_DIR / "test_allometry.csv"
    missing = tmp_path / "missing.csv"

    with pytest.raises(pydantic.ValidationError) as error:
        _with(valid, stand_params=_stand_params_reading(present, missing))

    assert str(missing) in str(error.value)
    assert str(present) not in str(error.value)

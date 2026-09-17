# Tests for CanopyLayerAllometry.with_single_allometry_per_layer
# and for the laziness of zones_data / zones_species_id.
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from susi.io.susi_parameter_model import (
    AllometryFileAndSpecies,
    CanopyLayerAllometry,
    CanopyLayerName,
)

# %% CanopyLayerAllometry.with_single_allometry_per_layer

# Fabricated AllometryFileAndSpecies values.
# No real CSV is ever read here.
allometry_file_and_species_strategy = st.builds(
    AllometryFileAndSpecies,
    file_path=st.text(min_size=1, max_size=20).map(lambda name: Path(f"{name}.csv")),
    species_id=st.integers(min_value=1, max_value=100),
)

layers_strategy = st.dictionaries(
    keys=st.sampled_from(list(CanopyLayerName)),
    values=allometry_file_and_species_strategy,
    max_size=len(CanopyLayerName),
)


@given(layers=layers_strategy, n=st.integers(min_value=0, max_value=50))
def test_with_single_allometry_per_layer_roundtrips(layers, n):
    result = CanopyLayerAllometry.with_single_allometry_per_layer(layers=layers, n=n)

    assert len(result.allometry_file_registry) == len(layers)

    for layer_name in CanopyLayerName:
        if layer_name not in layers:
            assert result.pointers[layer_name] is None
            continue

        layer_pointers = result.pointers[layer_name]
        assert layer_pointers is not None
        assert len(layer_pointers) == n

        # Every populated layer is homogeneous: its registry entry holds
        # exactly that layer's own AllometryFileAndSpecies, matching what a
        # hand-built CanopyLayerAllometry for the same data would look like.
        if n > 0:
            (registry_number,) = set(layer_pointers)
            assert result.allometry_file_registry[registry_number] == layers[layer_name]
        else:
            assert layers[layer_name] in result.allometry_file_registry.values()


def test_negative_n_silently_produces_empty_pointers():
    """n has no pydantic-enforced constraint on this classmethod (unlike
    species_id above, which is a real AllometryFileAndSpecies field) --
    passing a negative n does not raise. Python's `[x] * n` for n < 0
    evaluates to [], so a populated layer ends up with an empty pointer
    list rather than an error. Pinning this down since it's surprising:
    revisit if with_single_allometry_per_layer ever gains an explicit
    n >= 0 guard."""
    layers = {
        CanopyLayerName.dominant: AllometryFileAndSpecies(
            file_path=Path("pines.csv"), species_id=1
        )
    }

    result = CanopyLayerAllometry.with_single_allometry_per_layer(layers=layers, n=-1)

    assert result.pointers[CanopyLayerName.dominant] == []


# %% Lazy allometry-file parsing


def test_zones_data_not_read_until_accessed():
    registry = {
        1: AllometryFileAndSpecies(file_path=Path("does/not/exist.csv"), species_id=1)
    }
    pointers = {
        CanopyLayerName.dominant: [1],
        CanopyLayerName.subdominant: None,
        CanopyLayerName.under: None,
    }

    # Construction does no I/O, so a registry entry pointing at a
    # nonexistent CSV does not raise here.
    allometry = CanopyLayerAllometry(
        allometry_file_registry=registry, pointers=pointers
    )

    # zones_species_id never reads the file at all.
    assert allometry.zones_species_id == {1: 1}

    # zones_data does, and only raises once actually accessed.
    with pytest.raises(FileNotFoundError):
        allometry.zones_data

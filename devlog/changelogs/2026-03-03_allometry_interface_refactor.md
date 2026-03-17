# PR: Allometry Interface Refactor

## Summary

This PR refactors how allometry data flows through the system:

1. **Allometry data is now pre-loaded** in `SusiParams` during validation, rather than read on-demand from Excel files
2. **New `canopylayers` field** in `SiteParams` replaces implicit correspondence between canopy layers and allometry dictionary keys
3. **New validators** ensure stand ages and allometry pointers are consistent

## Files Changed

- `src/susi/io/susi_parameter_model.py`
- `src/susi/core/allometry.py`
- `src/susi/core/canopylayer.py`
- `src/susi/core/stand.py`
- `src/susi/core/susi_main.py`
- `tests/test_susi_model.py`
- `tests/data/test_allometry.xlsx`
- `tests/data/weather.csv`

<details>

## Detailed Changes

### Interface: MottiFileParams → AllometryParams

The parameter class was renamed and now auto-parses Excel files via `model_validator`:

```python
# Before
motti_file_parameters: MottiFileParams

# After
allometry_parameters: AllometryParams
```

The `path` field was renamed to `allometry_dir_path` for clarity.

Private cached data storage and exposed properties:
```python
_dominant_data: dict[int, pd.DataFrame]
_dominant_species_id: dict[int, int]
# ... same for subdominant and under
```

### New Field: canopylayers

Each `SiteParams` now requires a `CanopyLayerAllometryPointers` field:

```python
canopylayers: CanopyLayerAllometryPointers
```

Where:
```python
class CanopyLayerAllometryPointers(StrictFrozenModel):
    dominant: list[int]
    subdominant: list[int]
    under: list[int]
```

Each list must have length `L/2` (number of soil columns). Value `0` means "not in use". Non-zero values must correspond to keys in `AllometryParams.{layer}` dictionaries.

### Validators Added

1. **Stand age validation** (`stand_age_vs_allometry_pathway`): Ensures initial stand ages fall within allometry file ranges
   - Initial stand age >= minimum age in the allometry file
   - Initial stand age + simulation duration <= maximum age in the allometry file

2. **Canopy layer length** (`canopy_layer_elements`): Ensures pointer lists match soil column count
   ```python
   for layer_name in ["dominant", "subdominant", "under"]:
       layer_list = getattr(self.canopylayers, layer_name)
       if len(layer_list) != n:
           raise ValueError(f"length mismatch...")
   ```

3. **Pointer correspondence** (`allometry_files_pointers`): Ensures non-zero pointers reference valid allometry keys
   ```python
   non_zero_pointers = set(p for p in layer_pointers if p != 0)
   missing_keys = non_zero_pointers - layer_keys
   if missing_keys:
       raise ValueError(f"pointers {missing_keys} do not exist...")
   ```

### Allometry Class Refactored

Method signatures changed from file paths to pre-loaded DataFrames:

```python
# Before
def motti_development(self, ifile, sfc):
    # Reads file internally via self.get_motti(ifile)

# After
def motti_development(self, df, sp, sfc):
    # df: DataFrame from allometry Excel sheet 0
    # sp: species ID from allometry Excel sheet 1
    # sfc: site fertility class
```

The internal `get_motti` method was removed from the `Allometry` class.

### Stand Class Updated

Now passes `allometry_params` object instead of `mottifile`:

```python
# Before
Canopylayer(
    ...
    mottifile.path,
    mottifile.dominant,
    ...
)

# After
Canopylayer(
    ...
    allometry_params.dominant_data,
    allometry_params.dominant_species_id,
    ...
)
```

### Benefits

1. **Single-pass loading**: Allometry files are read once during `SusiParams` validation
2. **Caching**: The `@lru_cache` decorator ensures the same file isn't read multiple times
3. **Cleaner separation of concerns**: `Allometry` class focuses solely on interpolation logic
4. **Earlier error detection**: File loading errors are caught at parameter validation time
5. **Immutable by default**: Once validated, the allometry data cannot be modified

### Tests Added

- `test_valid_stand_age_all_layers` - Valid age within range
- `test_initial_dominant_age_below_minimum` - Raises error for too low age
- `test_initial_age_plus_duration_above_maximum` - Raises error for simulation exceeding max
- `test_subdominant_layer_validation` - Validates subdominant layer
- `test_under_layer_validation` - Validates understorey layer
- `test_valid_canopy_layer_pointers_length` - Valid pointers with correct length
- `test_invalid_canopy_layer_pointers_length` - Raises error for wrong length
- `test_valid_allometry_pointers_correspondence` - Valid pointer-to-key mapping
- `test_invalid_allometry_pointers_missing_key` - Raises error for invalid pointer

### Test Data

- `tests/data/test_allometry.xlsx` (copied from `src/inputs/SF_31.xlsx`)
- `tests/data/weather.csv` (copied from `src/inputs/weather/CFw.csv`)

</details>

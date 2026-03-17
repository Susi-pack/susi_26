# MottiFileParams Refactor Plan
## Overview

The Motti files are read too deep in the code. It is better practice to read them at the validation layer, before running anything.
The goal is to parse them first, then pass DataFrames around instead of paths pointing to files.

Additionally, we will restructure some of the code of the validation layer.

## Current State Problems

- Excel file reading happens deep in `allometry.py` (should be at validation layer)
- Validation layer has some repetition that will benefit from restructuring.
- canopylayers (see below) is harcoded and should not be.

## Current State Analysis

### MottiFileParams (old)
```python
class MottiFileParams(StrictFrozenModel):
    path: DirectoryPath
    dominant: dict[int, str]   # key=0: inactive, key>0: active → filename
    subdominant: dict[int, str]
    under: dict[int, str]
```

### SiteParams.canopylayers (old - hardcoded)
```python
@property
def canopylayers(self) -> dict[str, np.ndarray]:
    return {
        "dominant": np.ones(self.n, dtype=int),
        "subdominant": np.zeros(self.n, dtype=int),
        "under": np.zeros(self.n, dtype=int),
    }
```

### Current Code Path (file I/O deep in code):
```
SusiParams
  └─ MottiFileParams (stores path + dict[int,str])
       └─ susi_main.py passes mottifile to Stand
            └─ Stand passes mottifile.path + mottifile.dominant to Canopylayer
                 └─ Canopylayer: mfile = mottipath / mottifile[ncanopy]
                      └─ Canopylayer: self.allodic[ncanopy].motti_development(mfile, sfc)
                           └─ allometry.py: reads Excel file HERE
```

## New Design

### CanopyLayerAllometry (new)
```python
class CanopyLayerAllometry(StrictFrozenModel):
    allometry_files_dir: DirectoryPath
    filenames: list[str | None]  # None = no layer at this column.

    @model_validator(mode="after")
    def _validate_allometry_files(self) -> "CanopyLayerAllometry":
        # Validates file existence and Excel structure at construction time

    @property
    def allometry_dataframes(self) -> list[pd.DataFrame | None]:
        """Reads 1st page of excel allometry files for all soil columns"""

    @property
    def species_ids(self) -> list[int | None]:
        """Reads 2nd page of excel allometry files for all soil columns"""
```

### Helper Functions (in validation layer)
- `get_allometry_path_values_from_excel(filepath)` - reads sheet 0 (previously `get_motti()`)
- `read_species_id_from_allometry_file(filepath)` - reads species code from sheet 1
- `_basic_allometry_file_checks(filepath)` - validates file exists and has 2 sheets

### Validation Rules
- **File existence**: Validated in field validator - each non-None filename must point to an existing file
- **Excel structure**: Validated in property - sheet 0 must have 22 columns
- **Species code**: Given directly in class (no longer read from Excel sheet 1)
- **n validation**: At SiteParams level - `len(filenames) == n` must hold
- **Empty list**: Not allowed - all columns must have a value (None or str)

### Tests
Location: `tests/test_canopy_layer_motti.py`
- test_nonexistent_file_raises_error
- test_missing_sheet_raises_error (verify 2-sheet requirement for species ID)
- test_missing_columns_raises_error
- test_corrupt_excel_raises_exception
- test_wrong_length_raises_error (at SiteParams level)
- test_empty_string_raises_error
- test_valid_excel_parses_successfully
- test_species_ids_property
- test_golden (integration test with real Excel file)

### SiteParams (modified)
```python
class SiteParams(StrictFrozenModel):
    dominant_allometry: CanopyLayerAllometry
    subdominant_allometry: CanopyLayerAllometry
    under_allometry: CanopyLayerAllometry
    
    # Validator checks per_column length matches n
```

### New Code Path (DataFrames flow through)
```
SusiParams
  └─ SiteParams
       └─ CanopyLayerAllometry: species_code + filenames, parses Excel at validation
            └─ susi_main.py passes site_parameters to Stand
                 └─ Stand extracts unique files + their DataFrames
                      └─ Canopylayer receives DataFrames + species_code
                           └─ Canopylayer: self.allodic[ncanopy] = Allometry()
                                └─ Canopylayer: motti_development(df, sfc, species_code)
                                     └─ allometry.py: uses DataFrame directly (no file I/O)
```

## Step by step plan (TDD style)

### Phase 1: Create CanopyLayerAllometry Model
STATUS: COMPLETED

**Step 1: Create `CanopyLayerAllometry` class in `susi_parameter_model.py`**
- Add new class `CanopyLayerAllometry(StrictFrozenModel)` with:
  - `allometry_files_dir: DirectoryPath` - base directory for allometry files
  - `filenames: list[str | None]` - list of filenames (None = no layer at that column). Length equals n (number of soil columns). Different columns can use different Excel files.
- Use `@model_validator` for validation at construction time
- Use `@property` for `allometry_dataframes` and `species_ids` (pd.DataFrame is arbitrary type)
- Eager parsing: validate and parse Excel files immediately at construction time
- `species_ids` property reads species from Excel sheet 2

**Step 2: Add validation in `CanopyLayerAllometry`**
- `@model_validator` validates file existence and Excel structure at construction
- `@property allometry_dataframes` returns parsed DataFrames
- `@property species_ids` returns species codes from sheet 2

### Phase 2: Create Tests for CanopyLayerAllometry
STATUS: COMPLETED (all 8 tests PASS)

**Step 3: Create test file `tests/test_canopy_layer_motti.py`**
- `test_nonexistent_file_raises_error` - verify file existence validation (PASS)
- `test_missing_columns_raises_error` - verify 22 columns on sheet 0 (PASS)
- `test_corrupt_excel_raises_exception` - verify corrupt file handling (PASS)
- `test_empty_string_raises_error` - test empty strings are rejected (PASS)
- `test_valid_excel_parses_successfully` - test successful parsing (PASS)
- `test_species_ids_property` - test species_ids property (PASS)
- `test_golden` - integration test with real Excel file (PASS)
- `test_wrong_length_raises_error` - test length validation at SiteParams level (PASS)

### Phase 3: Integrate CanopyLayerAllometry into SiteParams
STATUS: COMPLETED

**Step 4: Modify `SiteParams` class**
- Add fields: `dominant_allometry: CanopyLayerAllometry`, `subdominant_allometry: CanopyLayerAllometry`, `under_allometry: CanopyLayerAllometry`
- Add validator to check `len(filenames) == n` for each layer
- Keep hardcoded `canopylayers` property unchanged

### Phase 4: Update SusiParams
STATUS: COMPLETED

**Step 5: Update `SusiParams` class**
- Remove `motti_file_parameters: MottiFileParams` field
- `SiteParams` now contains canopy layer data via `CanopyLayerAllometry` fields

### Phase 5: Update Stand and Canopylayer to use DataFrames
STATUS: COMPLETED

**Step 6: Modify `Stand` class (`stand.py`)**
- Change constructor to receive `site_parameters: SiteParams` instead of `mottifile: MottiFileParams`
- Extract `dominant_allometry`, `subdominant_allometry`, `under_allometry` from site_parameters
- Pass `allometry_dataframes` and `species_ids` to Canopylayer

**Step 7: Modify `Canopylayer` class (`canopylayer.py`)**
- Change constructor to receive `allometry_dataframes` and `species_ids` instead of `mottipath` and `mottifile`
- Use column indices from `ixs[ncanopy]` to get correct DataFrame for each layer
- Each `CanopyLayerAllometry` maps 1:1 to a `CanopyLayer`

### Phase 6: Update allometry.py to accept DataFrames
STATUS: COMPLETED

**Step 8: Modify `Allometry.motti_development` method**
- Modified to accept `(df, sfc, species)` instead of `(ifile, sfc)`
- Species is passed directly as integer (no longer read from Excel sheet 1)
- DataFrame is passed directly (already validated and parsed at validation layer)

### Phase 7: Update susi_main.py
STATUS: COMPLETED

**Step 10: Update `susi_main.py`**
- Pass `site_parameters` to `Stand` instead of `motti_file_parameters`
- Removed reference to old `MottiFileParams`

### Phase 8: Update Test Files and Scripts
STATUS: COMPLETED

**Step 11: Update test instantiation scripts**
- `src/scripts/mikko_susi_new.py` - Updated to use CanopyLayerAllometry
- `src/inputs/parameters/golden_test.py` - Updated to use CanopyLayerAllometry
- `tests/meaningless_susi_model.py` - Updated to use CanopyLayerAllometry

### Phase 9: Remove MottiFileParams
STATUS: COMPLETED

**Step 12: Remove `MottiFileParams` class**
- Deleted commented-out class from `susi_parameter_model.py`
- No remaining references to MottiFileParams

### Phase 10: Update all instances of SusiParams
STATUS: COMPLETED

**Step 13: Update all SusiParams instantiations**
- `tests/meaningless_susi_model.py` - Updated to use CanopyLayerAllometry
- `src/inputs/parameters/golden_test.py` - Updated to use CanopyLayerAllometry
- `src/scripts/mikko_susi_new.py` - Updated to use CanopyLayerAllometry
- `src/scripts/susi_parallel.py` - Uses golden_test.PARAMETERS as base (works via model_dump/validate)

Note: `src/scripts/susi_calls_scenarios.py` uses legacy API (not SusiParams) - out of scope

### Bug Fix: Canopylayer indexing fix
STATUS: COMPLETED

**Fixed indexing bug in `canopylayer.py`**:
- Changed `first_col_idx = self.ixs[ncanopy][0]` to `first_col_idx = self.ixs[ncanopy][0][0]`
- The original code returned a numpy array instead of a scalar integer, causing TypeError when indexing DataFrame list

# Still TODO / questions
- The mottifile has columns in Finnish, but they are immediately translated upon read into English. Does this make sense? Isn't it better to just name the cols in English?

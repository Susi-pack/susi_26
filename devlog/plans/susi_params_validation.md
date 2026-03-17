# Implementation Plan: Validation Logic for SusiParams

## Overview
Add two validation methods to ensure consistency between:
1. `CanopyLayerAllometryPointers` list lengths and the number of soil columns (`self.n`)
2. `CanopyLayerAllometryPointers` values and `AllometryParams` keys

## TODO 1: CanopyLayerAllometryPointers length validation (SiteParams)

**Location**: `src/susi/io/susi_parameter_model.py`, line ~688-690

**Task**: Complete the `canopy_layer_elements` model_validator in `SiteParams` to check that each list in `CanopyLayerAllometryPointers` has length equal to `self.n`.

**Logic**:
- `self.n = int(self.L / 2)` is the number of computation nodes
- Each of `canopylayers.dominant`, `canopylayers.subdominant`, `canopylayers.under` should have length equal to `self.n`
- Raise `ValueError` if lengths don't match

## TODO 2: Allometry files pointers validation (SusiParams)

**Location**: `src/susi/io/susi_parameter_model.py`, line ~743-746

**Task**: Complete the `allometry_files_pointers` model_validator in `SusiParams` to check one-to-one correspondence.

**Logic**:
- For each canopy layer (dominant, subdominant, under):
  - Get the set of non-zero values from `site_parameters.canopylayers.{layer}`
  - Get the set of keys from `allometry_parameters.{layer}`
  - Validate that all non-zero pointer values exist as keys in AllometryParams
  - Raise `ValueError` if there's a mismatch

## Tests to Add (test_susi_model.py)

### Test 1: Valid canopy layer pointers length
- Create SiteParams with `L=10.0` (so `n=5`)
- Set all three CanopyLayerAllometryPointers lists to have length 5
- Should pass validation

### Test 2: Invalid canopy layer pointers length
- Create SiteParams with `L=10.0` (so `n=5`)
- Set one of the CanopyLayerAllometryPointers lists to have wrong length
- Should raise ValueError with message about length mismatch

### Test 3: Valid allometry pointers correspondence
- All non-zero values in canopylayers point to valid keys in AllometryParams
- Should pass validation

### Test 4: Invalid allometry pointers - missing key
- Set canopylayers to reference a key that doesn't exist in AllometryParams
- Should raise ValueError about missing correspondence

### Test 5: Invalid allometry pointers - extra key
- Set AllometryParams to have keys that are never referenced
- Should raise value error about extra keys not being accepted

## Implementation Order
Stop development between phases to check with master.

Phase 1. Write failing tests first (TDD approach)

Phase 2. Implement validation in `SiteParams.canopy_layer_elements`

Phase 3. Implement validation in `SusiParams.allometry_files_pointers`; run tests to verify all pass

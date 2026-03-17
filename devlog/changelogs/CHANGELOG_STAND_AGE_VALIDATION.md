# Changelog: Stand Age Validation with Allometry Files

## Problem
The `SusiParams` model lacked validation to ensure that simulated stand ages do not exceed the maximum age available in the allometry Excel files. This could lead to interpolation errors or unexpected behavior when predicting stand attributes beyond the supported age range.

## Solution
Added validation in the `SusiParams` model (`_validate_stand_age_with_allometry` model validator) that:

1. Computes simulated years as `end_date.year - start_date.year`
2. For each canopy layer (dominant, subdominant, understorey), calculates the final predicted stand age as `initial_[layer]_stand_age_years + simulated_years`
3. Validates that this final age does not exceed the minimum of the maximum ages across all allometry DataFrames for that layer
4. Raises a clear `ValueError` if any layer exceeds its available age range

If a layer has no allometry files (all `None`), validation is skipped for that layer.

## Files Changed
- `src/susi/io/susi_parameter_model.py` - Added `_validate_stand_age_with_allometry` model validator (lines 714-751)
- `tests/test_susi_parameter_model.py` - New test file with 7 tests covering:
  - Valid case: final age below max age
  - Dominant layer violation: final age exceeds max
  - Subdominant layer violation: final age exceeds max
  - Under layer violation: final age exceeds max
  - Edge case: exact boundary (final age equals max)
  - All None allometry files (validation skipped)
  - Multiple soil columns (n=10)

## Test Results
All 7 tests pass, validating that the validation correctly catches age exceedances and allows valid configurations.

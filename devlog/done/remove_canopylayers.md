## Summary

Simplify the canopylayer handling by removing the redundant `SiteParams.canopylayers` property and deriving layer groups directly from the `CanopyLayerAllometry.filenames` list.

### Problem

After the mottifile → CanopyLayerAllometry refactor, the `canopylayers` property was redundant:
- **Old**: column index → canopylayer code → dict key → filename
- **Bug**: canopylayer codes didn't actually map to files - file selection was implicit based on column position

### Changes Made

1. **Removed `canopylayers` property** from `SiteParams` (`susi_parameter_model.py`)
2. **Updated `Stand.__init__`** to derive layer groups directly from filenames (`stand.py`)
3. **Updated call site** in `susi_main.py`
4. **Fixed bug** in `canopylayer.py` where `self.sfc` was being overwritten in the loop (only visible with multiple different allometry files)

### How it works now

Columns with the same filename share one allometry:
```python
filenames = ["pine.xlsx", "pine.xlsx", "spruce.xlsx", "birch.xlsx"]
# Creates 3 allometry groups: columns 0,1 share pine; column 2 is spruce; column 3 is birch
```

### Tests

Added comprehensive tests in `tests/test_stand.py`:
- `TestBuildLayerIndices` - 4 unit tests for layer index building
- `TestStandWithMultipleAllometryFiles` - 2 integration tests  
- `TestAllometryMappingToColumns` - 9 tests covering:
  - All columns same file
  - All columns different files
  - Non-contiguous sharing
  - Mixed None in middle
  - Subdominant/under layers
  - Correct dataframe reading (species IDs, ages)

### Benefits

1. Simpler API - no redundant canopylayers array
2. Correct behavior - grouping is explicit (same filename = same allometry)
3. Less code to maintain
4. Matches user intuition: filename at column i is used for column i


# Excel Caching Implementation Plan

## Status: COMPLETED

## Problem Statement

Currently, `CanopyLayerAllometry` reads Excel files multiple times:
1. Once in the `_validate_allometry_files` model validator (for each filename)
2. Once in the `allometry_dataframes` property (for each filename)
3. Once in the `species_ids` property (for each filename)

When `model_copy()` is called on `SusiParams` (e.g., to create 100 variants), each copy re-validates and re-reads all Excel files. For `golden_test.PARAMETERS` with 20 Excel files (`filenames=["CF_41.xlsx"] * 20`), this results in 2,000+ Excel file reads instead of 20.

## Solution

Add `@lru_cache` to the two Excel-reading functions:
- `get_allometry_path_values_from_excel(filepath)` - reads sheet 0
- `read_species_id_from_allometry_file(filepath)` - reads sheet 1

This ensures each unique file is read exactly once, regardless of how many times it's referenced or how many model copies are created.

## TDD Plan

### Phase 1: Failing Tests (demonstrate the problem)

**Test 1: `test_excel_read_count_with_caching`**
- Location: `tests/test_excel_caching.py` (new file)
- Purpose: Verify that multiple calls with same file result in single read
- Approach: Use `unittest.mock.patch` to count `pd.read_excel` calls
- Expected failure: Without caching, 3 calls to `get_allometry_path_values_from_excel` with same file results in 3 reads

**Test 2: `test_different_files_get_different_results`**
- Purpose: Verify cache key is file-path-sensitive
- Approach: Call function with two different Excel files, verify different results returned
- Expected failure: None (should pass even without caching, but validates correctness)

**Test 3: `test_cache_persists_across_model_copies`**
- Purpose: Verify caching works across Pydantic model copies
- Approach: Create `SusiParams`, call `model_copy()` 5 times, count actual Excel reads
- Expected failure: Without caching, 6x Excel reads (1 original + 5 copies)

### Phase 2: Implementation

**Step 1: Add `functools.lru_cache` to `get_allometry_path_values_from_excel`**
- File: `src/susi/io/susi_parameter_model.py`
- Change: Add `@lru_cache(maxsize=None)` decorator
- Note: Convert `Path` to `str` since `Path` is not hashable

**Step 2: Add `functools.lru_cache` to `read_species_id_from_allometry_file`**
- File: `src/susi/io/susi_parameter_model.py`
- Change: Add `@lru_cache(maxsize=None)` decorator
- Note: Convert `Path` to `str` since `Path` is not hashable

**Step 3: Verify import**
- Ensure `from functools import lru_cache` is present

### Phase 3: Passing Tests (verify solution)

**Test 4: `test_excel_read_count_with_caching` (should now pass)**
- Verify that 3 calls with same file result in exactly 1 read

**Test 5: `test_cache_persists_across_model_copies` (should now pass)**
- Verify that 5 `model_copy()` calls result in exactly 1 read per unique file

**Test 6: `test_100_simulation_params_efficient`**
- Purpose: Integration test replicating the original performance issue
- Approach: Create 100 `SimulationParams` using `golden_test.PARAMETERS.model_copy()`, measure time
- Expected: Should complete in <5 seconds (vs. minutes without caching)

### Implementation Notes

1. **No `.copy()` needed**: DataFrames are read-only; no mutation occurs in codebase
2. **Thread safety**: `lru_cache` is thread-safe; worst case is duplicate work on cache miss, not incorrect results
3. **Cache key**: String path (e.g., "/absolute/path/to/CF_41.xlsx") - different files get different cache entries
4. **Memory**: Each unique Excel file cached once as DataFrame; ~20 files = negligible memory

### Files to Modify

| File | Change |
|------|--------|
| `src/susi/io/susi_parameter_model.py` | Add `@lru_cache` decorators to 2 functions |
| `tests/test_excel_caching.py` | Create new test file with 6 tests |

### Timeline Estimate

- Phase 1 (failing tests): 30 minutes
- Phase 2 (implementation): 10 minutes  
- Phase 3 (passing tests): 20 minutes
- **Total**: ~1 hour

## Questions for Reviewer

1. Should the cache be cleared between test runs? (Currently: no, cache persists)
2. Should we add a `cache_clear()` method for testing purposes?
3. Is there a maximum cache size we should consider, or is unlimited (`maxsize=None`) acceptable?

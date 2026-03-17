# Plan: Further simplify Canopylayer - remove redundant nlyrs parameter

## Overview

This is a continuation of the canopylayer simplification. After removing the redundant `SiteParams.canopylayers` property and deriving layer groups from filenames, we can further simplify by removing the `nlyrs` parameter from Canopylayer.

## Current State

Currently, Canopylayer receives both:
- `nlyrs`: array of layer numbers (e.g., `[1, 2, 3]`)
- `ixs`: dict mapping layer number → column indices

This is redundant - `ixs.keys()` already contains the layer numbers.

## Changes Needed

### 1. Canopylayer.__init__ signature

**Before:**
```python
def __init__(
    self,
    name,
    nscens,
    yrs,
    ncols,
    nlyrs,          # <- remove this
    sfc,
    agearr,
    allometry_dataframes,
    species_ids,
    ixs,
    photopara,
    nut_stat,
):
```

**After:**
```python
def __init__(
    self,
    name,
    nscens,
    yrs,
    ncols,
    sfc,            # <- nlyrs removed, sfc moves up
    agearr,
    allometry_dataframes,
    species_ids,
    ixs,            # <- now use ixs.items() directly
    photopara,
    nut_stat,
):
```

### 2. Canopylayer.__init__ loop

**Before:**
```python
for ncanopy in nlyrs:
    if ncanopy > 0:
        layer_sfc = int(np.median(self.sfc[self.ixs[ncanopy]]))
        col_indices = self.ixs[ncanopy]
        ...
```

**After:**
```python
for layer_num, col_indices in ixs.items():
    layer_sfc = int(np.median(self.sfc[col_indices]))
    ...
```

### 3. Update all methods using nlyrs

- `initialize_domain()` - line ~89: remove `nlyrs = self.nlyrs`, use `self.ixs.items()`
- `update()` - line ~325: same
- `assimilate()` - line ~530: same  
- `cutting()` - line ~802: same

### 4. Stand.__init__ calls to Canopylayer

**Before:**
```python
ndominants = np.array(list(ixdominants.keys()))

self.dominant = Canopylayer(
    "dominant",
    n_scenarios,
    n_yrs,
    n_cols,
    ndominants,  # <- remove this
    sfc,
    ...
)
```

**After:**
```python
self.dominant = Canopylayer(
    "dominant",
    n_scenarios,
    n_yrs,
    n_cols,
    sfc,  # <- sfc moves up
    ...
)
```

## Benefits

1. **No more `if ncanopy > 0` checks** - layer 0 is just "no entry in dict"
2. **Cleaner iteration** - `for layer_num, col_indices in ixs.items()`
3. **Removes redundant parameter** - `nlyrs` is derived from `ixs.keys()`
4. **Single source of truth** - `ixs` contains all layer info

## Runtime Impact

Negligible - same O(n) complexity, just cleaner code.

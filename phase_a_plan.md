# Phase A — Basic JAX ops: Implementation Plan

## Goal

Convert every hot-path module to use `jax.numpy` instead of `numpy`, remove
in-place mutations, and replace linear algebra calls with JAX equivalents.
Each module stays **functionally identical** for numpy inputs — the change is
a mechanical translation that preserves numerics.

---

## Per-file conversion checklist

Every file follows this recipe **in order**:

### 1. Imports
```
- import numpy as np
+ import jax.numpy as jnp
```
```
+ from jaxtyping import Array
```
Remove `import scipy.linalg`, `import scipy.sparse` (replaced by JAX linalg).

### 2. Type annotations
```
- Float[np.ndarray, "shape"]
+ Float[Array, "shape"]
```
```
- np.ndarray           (in function signatures)
+ Array
```

### 3. Array creation
| Before | After |
|--------|-------|
| `np.zeros(s)` | `jnp.zeros(s)` |
| `np.ones(s)` | `jnp.ones(s)` |
| `np.full(s, v)` | `jnp.full(s, v)` |
| `np.zeros_like(a)` | `jnp.zeros_like(a)` |
| `np.full_like(a, v)` | `jnp.full_like(a, v)` |
| `np.linspace(a, b, n)` | `jnp.linspace(a, b, n)` |
| `np.arange(a, b, s)` | `jnp.arange(a, b, s)` |
| `np.tile(a, r)` | `jnp.tile(a, r)` |
| `np.array([...])` | `jnp.array([...])` |

### 4. Pure math operations
| Before | After |
|--------|-------|
| `np.exp(x)` | `jnp.exp(x)` |
| `np.log(x)` | `jnp.log(x)` |
| `np.maximum(a, b)` | `jnp.maximum(a, b)` |
| `np.minimum(a, b)` | `jnp.minimum(a, b)` |
| `np.where(c, a, b)` | `jnp.where(c, a, b)` |
| `np.abs(x)` | `jnp.abs(x)` |
| `np.clip(x, lo, hi)` | `jnp.clip(x, lo, hi)` |
| `np.square(x)` | `jnp.square(x)` |
| `np.sqrt(x)` | `jnp.sqrt(x)` |
| `np.mean(x)` | `jnp.mean(x)` |
| `np.sum(x)` | `jnp.sum(x)` |
| `np.cumsum(x)` | `jnp.cumsum(x)` |
| `np.flip(x)` | `jnp.flip(x)` |
| `np.divide(a, b)` | `a / b` (or `jnp.divide(a, b)`) |
| `np.ravel(x)` | `jnp.ravel(x)` |
| `np.reshape(x, s)` | `jnp.reshape(x, s)` |
| `np.max(x)` / `np.min(x)` (on arrays) | `jnp.max(x)` / `jnp.min(x)` |
| `max(x)` / `min(x)` (Python built-in on arrays) | `jnp.max(x)` / `jnp.min(x)` |
| `np.finfo(x)` | `jnp.finfo(x)` |
| `np.pi` | `jnp.pi` |

### 5. Remove `.copy()` calls
```
- arr.copy()
+ arr
```
JAX arrays are immutable — copies are unnecessary.

### 6. In-place mutations → `.at[].set()`

| Pattern | Replacement |
|---------|-------------|
| `arr[i] = x` | `arr.at[i].set(x)` |
| `arr[i, j] = x` | `arr.at[i, j].set(x)` |
| `arr[i] += x` | `arr.at[i].add(x)` |
| `arr[i, j] += x` | `arr.at[i, j].add(x)` |
| `arr[ixs] = x` (slice) | `arr.at[ixs].set(x)` |
| `arr[i == j] = x` | Use `jnp.eye(n) * diag` or `jnp.where(mask, x, arr)` |
| `arr.fill(0)` | Remove entirely (build fresh array instead) |
| `arr[mask] = x` (boolean mask) | `arr = jnp.where(mask, x, arr)` |
| `arr[mask] = op(arr[mask])` | `arr = jnp.where(mask, op(jnp.where(mask, arr, 0)), arr)` |
| `u[:] = result` | `u = result` (rebind instead of mutate) |

### 7. Linear algebra
```
- np.linalg.multi_dot([np.linalg.inv(A), hs])
+ jnp.linalg.solve(A, hs)
```
```
- scipy.linalg.solve(A, b, assume_a='tridiagonal')
+ jax.lax.linalg.tridiagonal_solve(dl, d, du, b)
```
```
- scipy.sparse.diags(diagonals, offsets, shape) @ vec
+ sum(jnp.eye(shape[0], k=offset) * diag for offset, diag in zip(offsets, diagonals)) @ vec
  # Or for tridiagonal specifically (full-matrix fallback, not preferred):
  jnp.eye(n) * diag + jnp.eye(n, k=1) * upper + jnp.eye(n, k=-1) * lower
```

### 8. `np.gradient` → `jnp.gradient`

`jnp.gradient` exists in JAX ≥ 0.4 (confirmed in 0.10). Use `jnp.gradient` directly — no custom helper needed.

---

## Implementation order by file

### ✓ Batch 1 — Leaf modules (done)

| # | File | Scope | Status |
|---|------|-------|--------|
| 1 | `fertilization_models/no_fertilization.py` | 1 `np.zeros` call | ✅ |
| 2 | `fertilization_models/npk.py` | 5 calls: `np.exp` (×3), `np.ones` (×1), `np.ndarray` annotation | ✅ |
| 3 | `fertilization_models/ash.py` | 13 calls + 2 loop-indexed mutations | ✅ |
| 4 | `mosslayer.py` | 32 calls + 3 `.copy()` (already functional style) | ✅ |

### ✓ Batch 2a — Temperature tridiagonal refactor (done)

| # | File | Scope | Status |
|---|------|-------|--------|
| 5 | `temperature.py` | Replace `scipy.linalg.solve(A, b, assume_a='tridiagonal')` with `jax.lax.linalg.tridiagonal_solve(dl, d, du, b)`. Store three diagonals (`d`, `dl`, `du`) in `ComputedConstants` instead of full matrix `A`. This is a structural change to the dataclass. | ✅ |

### ✓ Batch 2b — Mechanical np→jnp for medium-complexity modules (done)

| # | File | Scope | Status |
|---|------|-------|--------|
| 6 | `temperature.py` | Remaining `np.*` → `jnp.*`, `Float[np.ndarray]` → `Float[Array]`, in-place mutations → `.at[].set()` | ✅ |
| 7 | `gvegetation.py` | 74 calls + heavy indexed-assignment mutation + `_fill_site_nutrients` refactor | ✅ |
| 8 | `canopygrid.py` | 145 calls + heavy boolean-indexed mutation + 1 `.copy()` | ✅ |

### ✓ Batch 3 — High complexity: mutation + linalg (done)

| # | File | Scope | Status |
|---|------|-------|--------|
| 9 | `strip.py` | 83 calls + `np.linalg.multi_dot([inv(A), hs])` → `jax.lax.linalg.tridiagonal_solve` + 6 `.copy()` removed + `NumericalBuffer` removal. `_amatrix` and `_bound_const` eliminated — diagonals built inline. Uses `jnp.gradient` (exists in JAX 0.10, no custom helper). | ✅ |
| 10 | `susi_main.py` | Remove `make_numerical_buffer` call + `buffer` parameter propagation (3 sites) | ✅ |

**Tests written before converting:**
- `tests/core/test_strip.py`: `test_right_side`, `test_gmean_tr`, `test_hadjacent`, `test_runoff`, `test_compute_exfil`, `test_run_timestep`

### Batch 4 — High complexity: sparse linalg

| # | File | Scope |
|---|------|-------|
| 11 | `esom.py` | 128 calls + `scipy.sparse.diags` + 4 `.copy()` |

**Tests to write before converting:**
- `tests/core/test_esom.py`: `test_decompose` (kinetic matrix assembly)

### Batch 5 — Large files

| # | File | Scope |
|---|------|-------|
| 12 | `stand.py` | 183 calls + 3 `.copy()` + heavy in-place in `Stand` class |
| 13 | `canopylayer.py` | 264 calls + 8 `.copy()` + heaviest mutation |
| 14 | `susi_utils.py` | 242 calls + 2 `.copy()` + heavy in-place |

**Tests to write before converting:**
- `tests/core/test_stand.py`: augment existing with pure-function tests
- `tests/core/test_canopylayer.py`: augment existing with pure-function tests
- `tests/core/test_susi_utils.py`: key utility functions

---

## Cross-module dependency graph

```
Batch 1 (leaf, done)
  fertilization_models/*.py     (no deps)
  mosslayer.py                   ─┐
                                  │ strip.py imports from mosslayer
Batch 2a (tridiagonal refactor)
  temperature.py                  │  (structural: ComputedConstants changes)
                                  │
Batch 2b (mechanical np→jnp)
  gvegetation.py                  │
  temperature.py                  │  (remaining mechanical changes)
  canopygrid.py                   │
                                  │
Batch 3                           │
  strip.py              ←────────┘  (imports mosslayer.ReturnflowOutputs, temperature changes)
  → also affects susi_main.py      (imports strip.run_timestep, strip.make_numerical_buffer)
                                  │
Batch 4                           │
  esom.py               ←────────┘ (imports strip.ResidenceTimeOutput, susi_utils.peat_hydrol_properties)
                                  │
Batch 5                           │
  stand.py              ←────────┘ (imports from susi_utils, strip, temperature, esom, mosslayer, etc.)
  canopylayer.py                  │
  susi_utils.py          ←────────┘  (leaf-ish but referenced by everyone)
```

**Ordering constraint:** `mosslayer.py` must precede `strip.py`. `susi_utils.py` must precede `esom.py` (and `strip.py`). Within a batch, order doesn't matter except for `susi_utils.py` which should go last in batch 5 because most modules import from it. Batch 2a (temperature tridiagonal refactor) should precede Batch 5 (stand.py imports temperature).

---

## Verification strategy

### Per-file (before/after)

For each file, before making changes:

1. **Identify key public functions** that have well-defined inputs/outputs.
2. **Write a pytest test** that calls the function with known numpy inputs and asserts on expected outputs.
3. **Run the test** to confirm it passes with the original code.
4. **Apply the Phase A changes** to the module.
5. **Run the test again** — it must still pass (same input → same output, modulo array type).

### Integration

After all conversions are done:
```bash
pytest tests/ -x --tb=short
```

### Type checking
```bash
ty src/supersusi/core/
```
(Note: some files are excluded from ty in `pyproject.toml` — verify the coverage.)

---

## Per-file detailed notes

### `fertilization_models/no_fertilization.py`
- Only change: `np.zeros(n_cols)` → `jnp.zeros(n_cols)`, `np.ndarray` → `Array`

### `fertilization_models/npk.py`
- `np.exp` (×3) → `jnp.exp`, `np.ones` → `jnp.ones`, `np.ndarray` → `Array`

### `fertilization_models/ash.py`
- `np.arange`, `np.zeros`, `np.pi`, `np.ones` → `jnp.*`
- Lines 82-91: `K_release_history[i] = dK/dt` → build via `jnp.array([...])` or use `range` and functional accumulation

### `mosslayer.py`
- 32 `np.*` → `jnp.*`, 3 `.copy()` removal, `np.ndarray` → `Array`
- Lines 126-132: Currently uses `-=` on locals (`potinf`, `Wsto_top`). These are local rebinds, not array mutations — safe to keep as-is (just convert `np.maximum`/`np.minimum`).

### `gvegetation.py`
- 74 `np.*` → `jnp.*`, `np.ndarray` → `Array`
- Heavy pattern: `arr = np.zeros(n)`, then `arr[ix] = value`. Convert to `jnp.zeros(n).at[ix].set(value)`.
- Multiple-index pattern: chain `.at[]` calls or build with `jnp.where` for boolean masks.

### `temperature.py` (Batch 2a — tridiagonal refactor)
- **Structural change**: Replace `scipy.linalg.solve(A, b, assume_a='tridiagonal')` with `jax.lax.linalg.tridiagonal_solve(dl, d, du, b)`.
- Instead of full matrix `A`, store three diagonals in `ComputedConstants`: `d` (main), `dl` (lower), `du` (upper). Remove `A` field.
- `_create_linear_system_matrix`: replace with `_create_tridiagonal_system` that returns `(dl, d, du, b)` tuple.
- `run_timestep`: replace `u[:] = linalg.solve(A, b)` with `T_soil = tridiagonal_solve(dl, d, du, b)`, drop the `u` preallocation.

### `temperature.py` (Batch 2b — mechanical changes)
- Remaining `np.*` → `jnp.*`, `Float[np.ndarray]` → `Float[Array]`, `np.ndarray` → `Array`
- `b[0] = T_air` → `b.at[0].set(T_air)`
- `b[-1] = T_air_mean` → `b.at[-1].set(T_air_mean)`
- `np.max` → `jnp.maximum` (the `max(-5.0, inputs.T_air)` is a Python built-in on a float — keep as-is; it's for scalar conditionals that are Phase D)

### `canopygrid.py`
- 145 `np.*` → `jnp.*`, `np.ndarray` → `Array`
- **Boolean mask mutation is the main pattern:**
  ```python
  # Old:
  fW[T >= Tmax] = 1.0
  fS[T <= Tmin] = 1.0
  fW[ix] = calc(...)
  fS[ix] = 1.0 - fW[ix]
  
  # New:
  fW = jnp.where(T >= Tmax, 1.0, fW)
  fS = jnp.where(T <= Tmin, 1.0, fS)
  fW = fW.at[ix].set(calc(...))
  fS = fS.at[ix].set(1.0 - fW[ix])  # order matters: use the updated fW
  ```
- `np.shape(T) != gridshape` → not touched (Phase D)
- `W.copy()` → `W`

### `strip.py`
- 83 `np.*` → `jnp.*`, `np.ndarray` → `Array`, `Callable[[np.ndarray], np.ndarray]` → `Callable[[Array], Array]`
- **`_amatrix`** — **Eliminated entirely.** The full n×n matrix is never built. Instead, the tridiagonal system's three diagonals are computed inline in `run_timestep`:
  ```python
  d = implic * (Trwest + Treast) + alfa          # main diag,  length n
  du = -implic * Trwest                           # upper diag, length n (du[-1] ignored)
  dl = -implic * Treast                           # lower diag, length n (dl[0] ignored)
  d = d.at[0].set(1.0).at[-1].set(1.0)            # Dirichlet BCs
  du = du.at[0].set(0.0)                         # A[0,1] = 0
  dl = dl.at[-1].set(0.0)                        # A[-1,-2] = 0
  ```
- **`_bound_const`** — **Eliminated.** BCs are applied directly on the diagonals above.
- **`_right_side`**: `hs[0] = ...` / `hs[n-1] = ...` → `.at[0].set(...)` / `.at[n-1].set(...)`. The `if not DrIrr:` conditional remains for Phase D.
- **`compute_exfil`**: `dwt[0] = h0ts_west` → `.at[0].set(h0ts_west)`
- **`run_timestep`**:
  - Remove `NumericalBuffer` parameter
  - Remove `A = buffer.A; A.fill(0)` — build diagonals inline instead
  - Remove `A = _amatrix(...)` / `A = _bound_const(...)` calls
  - `Htmp = state.H.copy()` → `Htmp = state.H`
  - `Htmp1 = state.H.copy()` → `Htmp1 = state.H`
  - `dwt[0] = ...` → `.at[0].set(...)`
  - `np.linalg.multi_dot([np.linalg.inv(A), hs])` → `jax.lax.linalg.tridiagonal_solve(dl, d, du, hs[:, None])[:, 0]`
  - `conv = max(np.abs(Htmp1 - Htmp))` → `conv = jnp.max(jnp.abs(Htmp1 - Htmp))`
  - `if conv < 1.0e-7: break` — not touched (Phase D)
  - Allocate diagonals locally each iteration (O(n) memory instead of O(n²))
- **`compute_residence_time`**: `np.gradient(H, dist)` → `jnp.gradient(H, dist)` (exists in JAX 0.10). `timetoditch[ixwest] = ...` → `.at[ixwest].set(...)`.
- **Buffer removal in `susi_main.py`**: Remove `strip.make_numerical_buffer(params.strip)` call, remove `buffer` parameter from both `_run_daily_step` and `_run_annual_step`, update all call sites.

### `esom.py`
- 128 `np.*` → `jnp.*`, `np.ndarray` → `Array`
- Lines 541-556: Replace `scipy.sparse.diags` with dense construction:
  ```python
  # kmat = diags(diagonals, offsets, shape) @ M_tmp
  kmat = sum(
      jnp.eye(length, k=offset) * jnp.pad(diag, (max(0, -offset), max(0, offset)))[:length]
      for offset, diag in zip(offsets, diagonals)
  )
  # Actually, for this 9-diagonal case, just construct the bands directly:
  n = length
  A = jnp.zeros((n, n))
  A = A.at[jnp.arange(n), jnp.arange(n)].set(k_diag)
  A = A.at[jnp.arange(1, n), jnp.arange(n - 1)].set(k_low0[1:])
  # ... etc for each offset
  M_tmp = kmat @ M_tmp
  ```
  Actually, the simplest JAX-compatible approach is to use `jnp.diag`:
  ```python
  kmat = jnp.diag(k_diag)
  kmat += jnp.diag(k_low0[1:], k=-1)
  kmat += jnp.diag(k_low1[2:], k=-2)
  # ... etc
  ```
- Lines 491-537: Heavy indexed assignment for `k_diag[:, :, i] = ...` — these are 3D arrays. Convert to `.at[:, :, i].set()`.
- `M[:, :, 2] = LL_mass` → `M.at[:, :, 2].set(LL_mass)`
- `np.gradient(daily_cumulative_out[y, :])` → `jnp.gradient`
- 4 `.copy()` calls → remove

### `stand.py`
- 183 `np.*` → `jnp.*`, `np.ndarray` → `Array`
- The `Stand` class (line 545) is mutable OOP with `self.*` assignments — **convert only the pure dataclass-based functions** in this phase. Leave `Stand` class untouched for now.
- Pure functions like `_merge_dom_sub_under`, `_compute_lai` need mutation conversion.
- `laiout[layer] = ...` → `.at[layer].set(...)`
- `nstat[0, :] = ...` → `.at[0, :].set(...)`
- 3 `.copy()` calls → remove

### `canopylayer.py`
- 264 `np.*` → `jnp.*`, `np.ndarray` → `Array`
- Largest file with heaviest mutation. Same approach as stand.py: focus on pure dataclass-based functions, leave `Canopylayer` class untouched.
- 8 `.copy()` calls → remove
- Indexed assignments: convert to `.at[].set()` chains or `jnp.where`

### `susi_utils.py`
- 242 `np.*` → `jnp.*`, `np.ndarray` → `Array`
- Heavy in-place mutation patterns throughout
- `np.gradient` usages (lines 205, 886-888, 1124-1126) → `jnp.gradient`

---

## Files NOT in scope for Phase A

| File | Reason |
|------|--------|
| `susi_main.py` (except buffer removal) | Orchestration layer; convert after core modules are done |
| `allometry.py` | Not in hot-path per Phase A list |
| `stem_curve.py` | Not in hot-path per Phase A list |
| `thinning_models.py` | Not in hot-path per Phase A list |
| `weibull_recovery.py` | Not in hot-path per Phase A list |
| `metsi/*.py` | Not in hot-path per Phase A list |
| `io/*.py` | I/O layer, not hot-path |

## After Phase A completes

The code will:
- Use `jnp` everywhere (numpy arrays accepted, JAX arrays returned)
- Have no `.copy()` calls
- Have no in-place array mutations
- Use `jax.lax.linalg.tridiagonal_solve` instead of `np.linalg.multi_dot([inv(A), hs])` (strip) and `scipy.linalg.solve(A, b, assume_a='tridiagonal')` (temperature)
- Use dense `jnp.diag` construction instead of `scipy.sparse.diags`
- Use `jnp.max(jnp.abs(x))` instead of `max(np.abs(x))`
- No longer create/pass `NumericalBuffer` in strip or susi_main
- Use `jnp.gradient` instead of `np.gradient` (exists in JAX ≥ 0.4)

It will **still NOT be `jax.grad`-ready** because:
- `strip.run_timestep` has `if conv < 1e-7: break` on tracer (Phase D1)
- `strip._right_side` has `if Htmp1[0] > Htmp1[1]` on tracer (Phase D2)
- `temperature.run_timestep` has `if swe > 0.01` on tracer (Phase D3)
- `canopygrid._canopy_water_snow` has `if np.shape(T) != gridshape` (Phase D4)
- `strip.ComputedConstants` has `Callable` (interp1d) fields that choke on tracers (Phase B1)
- No PyTree registration on dataclasses (Phase E)

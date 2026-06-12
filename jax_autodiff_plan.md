# JAX Autodiff Plan — Operation-Level Compatibility

## Goal

Make the existing Python-loop code autodiff-able through `jax.grad`. The loops
(annual, daily, solver iterations) stay as Python `for` — they unroll at trace
time since bounds are concrete. What we fix is every **operation** that chokes
on a JAX tracer.

---

## What blocks `jax.grad` today

Even with `params` traced and loops unrolled, the tracer cannot flow through:

| Issue | Where | Failure |
|---|---|---|
| `np.linalg.inv` / `np.linalg.multi_dot` | `strip.run_timestep` | numpy LAPACK on a tracer |
| `scipy.linalg.solve` | `temperature.run_timestep` | scipy function on a tracer |
| In-place mutation (`A[i==j]=`, `hs[0]=`, `b[0]=`, `.fill()`) | strip, temperature | JAX arrays are immutable |
| `if conv < 1e-7: break` on tracer | `strip.run_timestep` | Python control flow on tracer |
| `if swe > 0.01` / `if Htmp1[0] > Htmp1[1]` on tracer | temperature, strip | Python `if` on tracer |
| `np.maximum`, `np.where`, `np.abs`, `max()` on tracer | everywhere | numpy functions on tracer |
| `scipy.interpolate.interp1d(tracer)` | strip, esom, canopylayer | scipy on tracer |
| `pd.DataFrame.loc[tracer]` / `.mean()` on tracer | methane, stand, susi_utils | pandas on tracer |
| opaque `photopara: Any` | stand, canopylayer | can't flatten in pytree |

---

## Phase A — Basic JAX ops

### A1. `np` → `jnp` everywhere in hot-path modules

Modules: `mosslayer.py`, `canopygrid.py`, `strip.py`, `temperature.py`, `stand.py`,
`canopylayer.py`, `esom.py`, `gvegetation.py`, `methane.py`, `susi_utils.py`,
`fertilization.py`, `fertilization_models/*.py`.

- `import numpy as np` → `import jax.numpy as jnp`
- `np.zeros` → `jnp.zeros`, `np.ones` → `jnp.ones`, `np.exp` → `jnp.exp`,
  `np.log` → `jnp.log`, `np.maximum` → `jnp.maximum`, `np.minimum` → `jnp.minimum`,
  `np.where` → `jnp.where`, `np.abs` → `jnp.abs`, `np.clip` → `jnp.clip`,
  `np.square` → `jnp.square`, `np.mean` → `jnp.mean`, `np.sum` → `jnp.sum`,
  `np.cumsum` → `jnp.cumsum`, `np.flip` → `jnp.flip`, `np.gradient` → custom or `jnp.diff`,
  `np.full` → `jnp.full`, `np.zeros_like` → `jnp.zeros_like`, `np.full_like` → `jnp.full_like`,
  `np.finfo` → `jnp.finfo`, `max(x)` → `jnp.max(x)`, `min(x)` → `jnp.min(x)`.

### A2. Remove `.copy()` calls

JAX arrays are immutable — no copy needed. `arr.copy()` → `arr`.

### A3. In-place mutations → functional `.at[].set()`

**`strip.py`**:

| Location | Old | New |
|---|---|---|
| `run_timestep` L355–356 | `dwt[0] = h0ts_west; dwt[n-1] = h0ts_east` | `dwt.at[0].set(h0ts_west).at[n-1].set(h0ts_east)` |
| `run_timestep` L363 | `A.fill(0)` | Remove (build fresh, don't reuse buffer) |
| `_amatrix` L277–280 | `A[i==j] = ...; A[i==j+1] = ...; A[i==j-1] = ...` | Replace with tridiagonal construction: `jnp.eye(n) * diag + jnp.eye(n, k=1) * upper + jnp.eye(n, k=-1) * lower` |
| `_bound_const` L285–288 | `A[0,0]=1; A[0,1]=0; A[n-1,n-1]=1; A[n-1,n-2]=0` | `A.at[0].set([1, 0, ...]).at[n-1].set([..., 0, 1])` |
| `_right_side` L319–327 | `hs[0] = ...; hs[n-1] = ...` | `hs.at[0].set(...).at[n-1].set(...)` |
| `compute_exfil` L238–239 | `dwt[0] = ...; dwt[n-1] = ...` | `dwt.at[0].set(...).at[n-1].set(...)` |

**`temperature.py`**:

| Location | Old | New |
|---|---|---|
| `run_timestep` L102 | `b[0] = T_air` | `b.at[0].set(T_air)` |
| `run_timestep` L103 | `b[-1] = T_air_mean` | `b.at[-1].set(T_air_mean)` |

**`mosslayer.py`**:

| Location | Old | New |
|---|---|---|
| `run_interception` L119–121 | `Wsto_top_ini = state.Wsto_top.copy()` | `Wsto_top_ini = state.Wsto_top` |
| `run_interception` L123–132 | local var reassignments like `Wsto_top += interc` | Track mutations in new variables (already mostly functional — check L126–132) |

### A4. Linear algebra

- **`strip`** L389: `np.linalg.multi_dot([np.linalg.inv(A), hs])` → `jnp.linalg.solve(A, hs)`
- **`temperature`** L104: `scipy.linalg.solve(A, b, assume_a='tridiagonal')` → `jnp.linalg.solve(A, b)` (matrix is small, ~50×50)
- **`esom._decompose`**: `scipy.sparse.diags(...) @ array` → dense construction with `jnp.diag`/`jnp.eye` + `@`

---

## Phase B — `interp1d` → `jnp.interp` lookup tables

Replace every scipy interpolator in `ComputedConstants` with pre-computed
`(x_grid, y_grid)` arrays, then call `jnp.interp(query, x_grid, y_grid)`
at runtime.

### B1. `strip.ComputedConstants` — 6 interpolators from `CWTr`

Replace `dwtToSto: Callable`, `stoToGwl: Callable`, `dwtToTra: Callable`,
`C: Callable`, `dwtToRat: Callable`, `dwtToAfp: Callable` each with a pair
of arrays:

```python
@dataclass(frozen=True)
class ComputedConstants:
    ...
    dwtToSto_x: Float[Array, " n_grid"]
    dwtToSto_y: Float[Array, " n_grid"]
    stoToGwl_x: Float[Array, " n_grid"]
    stoToGwl_y: Float[Array, " n_grid"]
    dwtToTra_x: Float[Array, " n_grid"]
    dwtToTra_y: Float[Array, " n_grid"]
    C_x: Float[Array, " n_grid"]         # storage coefficient
    C_y: Float[Array, " n_grid"]
    dwtToRat_x: Float[Array, " n_grid"]
    dwtToRat_y: Float[Array, " n_grid"]
    dwtToAfp_x: Float[Array, " n_grid"]
    dwtToAfp_y: Float[Array, " n_grid"]
```

At runtime:
```python
Tr0 = jnp.interp(dwt, cc.dwtToTra_x, cc.dwtToTra_y)
```

`CWTr` in `susi_utils.py` already computes these on fixed grids (150 points
for sto/rat/afp, `nLyrs` points for transmissivity). The interp1d → array
conversion is straightforward.

### B2. `esom.SubstanceComputedConstants` — ~10 interpolators

Replace `wtToVfAir_top/middle/bottom`, `t2`–`t7`, `phi1236`, `phi4`, `phi5`
each with `(x_grid, y_grid)` pairs. These are all 1D monotonic curves evaluated
at runtime in `_get_rates`.

### B3. `canopylayer.ComputedConstants.allodic` — allometry interpolators

`AllometryFunctions` is a dataclass of ~18 `interp1d` per zone/species:
`ageToHdom`, `ageToBa`, `ageToVol`, `ageToYield`, `ageToBm`,
`bmToLeafMass`, `bmToLAI`, `bmToHdom`, `bmToYi`, `bmToBa`, `bmToLitter`,
`bmToStems`, `yiToVol`, `yiToBm`, `volToLogs`, `volToPulp`.

Replace each with `(x_vals, y_vals)` arrays. Pre-compute on a dense grid
(e.g., 1000 points × age range) and use `jnp.interp` at runtime.

### B4. `susi_utils.rew_drylimit(dwt)`

Currently creates a new `interp1d` on every call. Move the table to a
module-level constant:

```python
_WT_GRID = jnp.array([-150.0, -1.0, -0.5, -0.3, -0.1, 0.0])
_RE_GRID = jnp.array([0.0, 0.1, 0.4, 1.0, 1.0, 0.7])

def rew_drylimit(dwt):
    return jnp.interp(dwt, _WT_GRID, _RE_GRID)
```

### B5. `canopylayer._leaf_dynamics()`

Creates `interp1d` for N/P/K concentration curves and leaf longevity.
Move tables to `ComputedConstants` and pass them in.

---

## Phase C — Pandas → arrays in hot-path interfaces

### C1. `methane.py`

- `Inputs.dfwt: pd.DataFrame` → `Inputs.mean_wt: Float[Array, " n"]`
- `run_timestep`: remove `dfwt[date_slice].mean()`, accept pre-computed mean

### C2. `stand.py`

- `Inputs.forc: pd.DataFrame` → separate `Inputs.rg: Float[Array, " days n"]`,
  `vpd: ...`, `T: ...`
- `Inputs.wt: pd.DataFrame` → `Inputs.wt: Float[Array, " days n"]`
- `Inputs.afp: pd.DataFrame` → `Inputs.afp: Float[Array, " days n"]`

### C3. `canopylayer.py`

- `Inputs.forc: pd.DataFrame` → individual array fields (same as stand)

### C4. `susi_utils.assimilation_yr()`

- `dfforc: pd.DataFrame` → arrays `rg: Float[Array, " days"]`,
  `vpd: Float[Array, " days"]`, `Ta: Float[Array, " days"]`
- `wt: pd.DataFrame` → `Float[Array, " days n"]`

### C5. `susi_utils.heterotrophic_respiration_yr()`

- `dfwt: pd.DataFrame` + year string → `mean_wt: Float[Array, " n"]`
  (pre-computed May-Oct mean WT)
- `t5: pd.DataFrame` → `t5_mean: float`

### C6. `susi_utils.ojanen_2019()`

- Same pattern as C5.

### C7. `_run_annual_step` — `weather_data.loc[year]`

Pre-slice weather data into `AnnualForcing` arrays during
`_build_annual_forcings`. The annual step receives weather as arrays,
not as a `pd.DataFrame`.

---

## Phase D — Traced control flow (minimal patches)

### D1. `strip.run_timestep` — remove convergence break

```python
# old:
for _it in range(100):
    ...
    conv = max(np.abs(Htmp1 - Htmp))
    if conv < 1.0e-7:
        break

# new:
for _it in range(100):
    ...
# Always 100 iterations. Deferred: lax.while_loop for efficiency.
```

### D2. `strip._right_side` — `if Htmp1[0] > Htmp1[1]` → `jnp.where`

```python
# old:
hs[0] = Htmp1[1] if Htmp1[0] > Htmp1[1] else min(ele[0] + h0_west, Htmp1[1])

# new:
cond = Htmp1[0] > Htmp1[1]
hs = hs.at[0].set(
    jnp.where(cond, Htmp1[1], jnp.minimum(ele[0] + h0_west, Htmp1[1]))
)
```

### D3. `temperature.run_timestep` — `if swe > 0.01` → `jnp.where`

```python
# old:
if inputs.swe > 0.01:
    T_air = max(-5.0, inputs.T_air)
else:
    T_air = inputs.T_air + T_cool

# new:
T_air = jnp.where(
    inputs.swe > 0.01,
    jnp.maximum(-5.0, inputs.T_air),
    inputs.T_air + T_cool,
)
```

### D4. `canopygrid._canopy_water_snow` — remove shape guard

```python
# old:
if np.shape(T) != gridshape:
    T = np.ones(gridshape) * T
    ...

# new:
# Ensure broadcasting is correct statically at call site instead.
```

---

## Phase E — PyTree registration

Add `@jax.tree_util.register_dataclass` to every dataclass that flows
through autodiff:

**`susi_main.py`**: `ModuleParams`, `ModuleComputedConstants`,
`DailyState`, `AnnualState`, `DailyForcing`, `DailyOutputs`,
`AnnualForcing`, `AnnualOutputs`, `SimulationState`, `InitialOutputs`,
`SimulationOutput`.

**Per-module**: `Params`, `State`, `Outputs`, `Inputs`, `ComputedConstants`
(and sub-types like `ExfilOutputs`, `TimestepInputs`, `TimestepOutputs`,
`ReturnflowOutputs`, `InterceptionOutputs`, etc.).

**Pattern**:

```python
@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Foo:
    bar: Float[Array, " n"]
    baz: float
```

JAX handles frozen dataclasses automatically — each field becomes a
child in the pytree. Python scalars (`float`, `int`, `bool`, `str`)
become static metadata.

---

## Phase F — Opaque objects

### F1. `photopara: Any` in `stand.Inputs`

Trace through the code to find exactly which fields are accessed from
`photopara` (e.g., `gamma`, `tau`, `X0`, `Smax`, `kappa`, `alfa`, `nu`,
`beta`). Define a concrete dataclass:

```python
@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PhotoParameters:
    gamma: float
    tau: float
    X0: float
    Smax: float
    kappa: float
    alfa: float
    nu: float
    beta: float
```

Store in `ModuleComputedConstants` (not `stand.Inputs`).

### F2. `SusiParams.site_parameters` → `spara: Any`

`susi_utils.heterotrophic_respiration_yr` and `ojanen_2019` access
`spara.sfc` and `spara.bd_top`. Extract these as scalar/array fields
in `ModuleComputedConstants`.

---

## Effort & risk

| Phase | Files | Effort | Risk | Blocks autodiff? |
|---|---|---|---|---|
| A — jnp + in-place → functional | strip, temperature, esom, mosslayer | large | medium | ✅ immediate |
| B — interp1d → lookup tables | strip, esom, canopylayer, susi_utils | very large | high | ✅ immediate |
| C — pandas → arrays | methane, stand, canopylayer, susi_utils, susi_main | large | high | ✅ immediate |
| D — traced control flow | strip, temperature, canopygrid | small | low | ✅ immediate |
| E — PyTree registration | all modules | medium | low | needed for `jax.grad` output |
| F — opaque objects | stand, canopylayer, susi_main, su    si_utils | small | low | ✅ immediate |

## What stays unchanged (deferred)

| Feature | Why it's OK | When to revisit |
|---|---|---|
| Python `for forcing in forcings:` | concrete list → unrolls at trace time | When simulation length needs to be a tracer |
| Python `for dd in range(ndays):` | `ndays` is concrete (366 padded) | Memory of 50×366 unrolled steps |
| Python `if year.do_cutting:` | `year` is concrete from unrolled list | Never |
| `lax.scan` outer loop | Not needed for `jax.grad` | Large-scale optimization |
| `lax.while_loop` for strip | Removed break, always 100 iters | Performance optimization |
| `strip.NumericalBuffer` | Remove as part of Phase A3 refactor | Done as part of A3 |

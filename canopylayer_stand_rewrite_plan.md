# Canopylayer + Stand Functional Rewrite Plan

## Overview

Convert OOP `Canopylayer` (~40 mutable attributes, 6 methods) and `Stand` (~70 mutable attributes, 8 methods) into frozen-dataclass functional modules following SuperSUSI_functional_architecture.md.

### Key decisions

- `stand.py` is the **only module `susi_main.py` imports** for vegetation. It bundles 3 `canopylayer.Params` inside `stand.Params`, and 3 `canopylayer.ComputedConstants` inside `stand.ComputedConstants`. Nothing about `canopylayer` leaks into `susi_main`.
- `stand.State` nests `nut_stat` + 3 `canopylayer.State` instances — `susi_main` holds a single `stand_state` variable.
- The annual flow is a 2-step process:
  1. **`apply_allometry()`** — pure mapping from core state `(biomass, agearr, remaining_share)` → all derived allometric variables (stems, volume, litter, demands...). No side effects, always recomputable.
  2. **`grow_stand()`** — takes state + `apply_allometry` output, runs photosynthesis, leaf dynamics, computes biomass increment, updates state. Calls `apply_allometry` internally on the new state.
- `apply_allometry` and `grow_stand` are **both at the canopy-layer level** — the stand-level versions wrap them ×3 and add aggregation.
- One single `Outputs` dataclass at each level. No split between "growth outputs" and "allometry outputs".
- `assimilate_stand()` is the thick annual orchestrator: `grow_stand` ×3 → `_aggregate` → `update_nutrient_status` → optional `cut_stand`.
- TDD throughout: each implementation step starts with a failing test.

---

## 1. `canopylayer.py` — dataclasses

```python
@dataclass(frozen=True)
class Params:
    name: str = field(doc="'dominant' / 'subdominant' / 'under'")
    ncols: int
    nlyrs: np.ndarray = field(doc="unique allometry zone IDs in this layer")
    sfc: Float[np.ndarray, " ncols"] = field(doc="site fertility class per column")

@dataclass(frozen=True)
class ComputedConstants:
    allodic: dict[int, AllometryFunctions] = field(
        doc="per-zone allometry, built via build_allometry_interpolation_functions"
    )
    ixs: dict[int, np.ndarray] = field(doc="column indices per zone")
    tree_species: Int[np.ndarray, " ncols"] = field(doc="1=Pine, 2=Spruce, 3=Birch")

@dataclass(frozen=True)
class State:
    """Persistent per-layer state — the 3 variables that cross year boundaries."""
    agearr: Float[np.ndarray, " ncols"] = field(doc="years")
    biomass: Float[np.ndarray, " ncols"] = field(doc="kg/tree")
    remaining_share: Float[np.ndarray, " ncols"] = field(doc="thinning fraction 0..1")

@dataclass(frozen=True)
class Outputs:
    """All variables for this layer. Populated either by apply_allometry() or
    grow_stand(). The latter overrides growth-specific fields and cross-calculations
    (nonwoodylitter = finerootlitter + leaf_litter) on top of apply_allometry()."""

    # Allometric
    stems: Float[np.ndarray, " ncols"] = field(doc="trees/ha")
    basalarea: Float[np.ndarray, " ncols"] = field(doc="m2/tree")
    hdom: Float[np.ndarray, " ncols"] = field(doc="dominant height, m")
    Dg: Float[np.ndarray, " ncols"] = field(doc="mean diameter, cm")
    volume: Float[np.ndarray, " ncols"] = field(doc="m3/tree")
    leafarea: Float[np.ndarray, " ncols"] = field(doc="one-sided, m2/m2 per tree")
    leafmass: Float[np.ndarray, " ncols"] = field(doc="kg/tree")
    volumegrowth: Float[np.ndarray, " ncols"] = field(doc="m3/tree/yr")
    logvolume: Float[np.ndarray, " ncols"]
    pulpvolume: Float[np.ndarray, " ncols"]
    yi: Float[np.ndarray, " ncols"]

    # Photosynthesis
    NPP: Float[np.ndarray, " ncols"] = field(doc="kg/tree/yr")
    NPP_pot: Float[np.ndarray, " ncols"]

    # Leaf dynamics
    new_lmass: Float[np.ndarray, " ncols"]
    leaf_litter: Float[np.ndarray, " ncols"]
    C_consumption: Float[np.ndarray, " ncols"]
    Nleafdemand: Float[np.ndarray, " ncols"]
    Nleaf_litter: Float[np.ndarray, " ncols"]
    N_leaf: Float[np.ndarray, " ncols"]
    Pleafdemand: Float[np.ndarray, " ncols"]
    Pleaf_litter: Float[np.ndarray, " ncols"]
    P_leaf: Float[np.ndarray, " ncols"]
    Kleafdemand: Float[np.ndarray, " ncols"]
    Kleaf_litter: Float[np.ndarray, " ncols"]
    K_leaf: Float[np.ndarray, " ncols"]

    # Fine root litter
    finerootlitter: Float[np.ndarray, " ncols"]
    n_finerootlitter: Float[np.ndarray, " ncols"]
    p_finerootlitter: Float[np.ndarray, " ncols"]
    k_finerootlitter: Float[np.ndarray, " ncols"]

    # Nonwoody litter (allometry: = finerootlitter; grow_stand: = finerootlitter + leaf_litter)
    nonwoodylitter: Float[np.ndarray, " ncols"]
    n_nonwoodylitter: Float[np.ndarray, " ncols"]
    p_nonwoodylitter: Float[np.ndarray, " ncols"]
    k_nonwoodylitter: Float[np.ndarray, " ncols"]

    # Woody litter
    woodylitter: Float[np.ndarray, " ncols"]
    n_woodylitter: Float[np.ndarray, " ncols"]
    p_woodylitter: Float[np.ndarray, " ncols"]
    k_woodylitter: Float[np.ndarray, " ncols"]

    # Mortality litter
    woody_litter_mort: Float[np.ndarray, " ncols"]
    n_woody_litter_mort: Float[np.ndarray, " ncols"]
    p_woody_litter_mort: Float[np.ndarray, " ncols"]
    k_woody_litter_mort: Float[np.ndarray, " ncols"]
    non_woody_litter_mort: Float[np.ndarray, " ncols"]
    n_non_woody_litter_mort: Float[np.ndarray, " ncols"]
    p_non_woody_litter_mort: Float[np.ndarray, " ncols"]
    k_non_woody_litter_mort: Float[np.ndarray, " ncols"]

    # Nutrient demands
    n_demand: Float[np.ndarray, " ncols"]
    p_demand: Float[np.ndarray, " ncols"]
    k_demand: Float[np.ndarray, " ncols"]
    basNdemand: Float[np.ndarray, " ncols"]
    basPdemand: Float[np.ndarray, " ncols"]
    basKdemand: Float[np.ndarray, " ncols"]

@dataclass(frozen=True)
class CuttingOutputs:
    """Only non-zero in cutting years."""
    harvested_volume: Float[np.ndarray, " ncols"]
    harvested_log_volume: Float[np.ndarray, " ncols"]
    harvested_pulp_volume: Float[np.ndarray, " ncols"]
    harvested_biomass: Float[np.ndarray, " ncols"]
    harvested_stems: Float[np.ndarray, " ncols"]
    nonwoody_lresid: Float[np.ndarray, " ncols"]
    n_nonwoody_lresid: Float[np.ndarray, " ncols"]
    p_nonwoody_lresid: Float[np.ndarray, " ncols"]
    k_nonwoody_lresid: Float[np.ndarray, " ncols"]
    woody_lresid: Float[np.ndarray, " ncols"]
    n_woody_lresid: Float[np.ndarray, " ncols"]
    p_woody_lresid: Float[np.ndarray, " ncols"]
    k_woody_lresid: Float[np.ndarray, " ncols"]
```

---

## 2. `canopylayer.py` — functions

### `apply_allometry`

```python
def apply_allometry(
    biomass: Float[np.ndarray, " ncols"],
    agearr: Float[np.ndarray, " ncols"],
    remaining_share: Float[np.ndarray, " ncols"],
    cc: ComputedConstants,
) -> Outputs
```

Pure. Replaces `update()`. Maps core state to all derived variables via allometric interpolation functions. Growth fields (NPP, leaf_litter, etc.) are set to 0. `nonwoodylitter = finerootlitter` (leaf_litter unknown — will be overridden by `grow_stand`).

For each zone in `cc.allodic`:
- `stems = biomass_to_stand.stems(bm) * remaining_share`
- `basalarea`, `hdom`, `Dg`, `leafarea`, `leafmass`, `volume` from `biomass_to_stand`
- `n_demand`, `p_demand`, `k_demand` from `nutrient_demand`
- Fine root litter, woody litter, mortality, logging residues from respective sub-modules

### `grow_stand`

```python
def grow_stand(
    state: State,
    cc: ComputedConstants,
    photopara,
    forc: pd.DataFrame,
    wt: pd.DataFrame,
    afp: pd.DataFrame,
    previous_nut_stat: Float[np.ndarray, " ncols"],
    nut_stat: Float[np.ndarray, " ncols"],
    lai_above: Float[np.ndarray, " ncols"],
) -> tuple[State, Outputs]
```

The annual growth computation. Body:

```python
# 1. Apply allometry on current state (gives stems, leafarea, finerootlitter, woodylitter)
allom = apply_allometry(state.biomass, state.agearr, state.remaining_share, cc)

# 2. Photosynthesis
LAI_photo = allom.leafarea * 2 * allom.stems          # double-sided
npp, npp_pot = assimilation_yr(photopara, forc, wt, afp, LAI_photo, lai_above * 2)

# Convert to tree basis (handle zero stems)
mask = allom.stems > 0
npp = np.divide(npp * nut_stat, allom.stems, out=np.full_like(npp, np.nan), where=mask)
npp_pot = np.divide(npp_pot * nut_stat, allom.stems, out=np.full_like(npp_pot, np.nan), where=mask)

# 3. Leaf dynamics
(new_lmass, leaf_litter, C_consumption, ...) = _leaf_dynamics(
    state.biomass, npp, allom.leafmass,
    previous_nut_stat, nut_stat, state.agearr,
    cc.allodic, cc.tree_species,
)

# 4. Biomass increment (leaf dynamics allocates carbon to leaves; rest goes to stem)
delta_bm = npp - C_consumption - allom.finerootlitter - allom.woodylitter
new_biomass = state.biomass + np.maximum(delta_bm, 0)
new_agearr = state.agearr + 1
new_state = State(new_agearr, new_biomass, state.remaining_share)

# 5. Re-apply allometry on new state
new_allom = apply_allometry(new_biomass, new_agearr, state.remaining_share, cc)

# 6. Merge: start from new allometry, override growth fields + cross-calculations
outputs = dataclasses.replace(
    new_allom,
    NPP=npp,
    NPP_pot=npp_pot,
    new_lmass=new_lmass,
    leaf_litter=leaf_litter,
    C_consumption=C_consumption,
    ...  # all leaf dynamics fields
    nonwoodylitter=new_allom.finerootlitter + leaf_litter,
    n_nonwoodylitter=new_allom.n_finerootlitter + Nleaf_litter,
    ...  # same for P, K
)
return new_state, outputs
```

### `_leaf_dynamics`

```python
def _leaf_dynamics(
    bm, bm_increment, current_leafmass, previous_nut_stat, nut_stat,
    agenow, allometry_funcs, species, printOpt=False
) -> tuple[...]
```

Already structurally pure — just extract as module-level private function. No changes to the computation.

### `compute_constants`

```python
def compute_constants(
    params: Params,
    allometry_df: pd.DataFrame,
    species_id: int,
) -> ComputedConstants
```

For each non-zero zone in `params.nlyrs`:
- Call `build_allometry_interpolation_functions(AllometryParams(species, sfc_median), allometry_df)`
- Build `tree_species` per column from species per zone
Return `ComputedConstants(allodic, ixs, tree_species)`.

### `compute_initial_state`

```python
def compute_initial_state(
    params: Params,
    cc: ComputedConstants,
    agearr: Float[np.ndarray, " ncols"],
    nut_stat: Float[np.ndarray, " ncols"],
) -> tuple[State, Outputs]
```

Replaces `initialize_domain()`. For each zone, set initial values from age:
- `biomass = age_based.bm(agearr)`
- `stems = biomass_to_stand.stems(biomass) * 1.0`
- `basalarea = age_based.ba(agearr)`, `hdom`, `Dg`, `leafarea`, `leafmass`, `volume`, demands from allometry
Return `State(agearr, biomass, remaining_share=1.0)` + `apply_allometry(biomass, agearr, 1.0, cc)`.

### `cut_stand`

```python
def cut_stand(
    state: State,
    cc: ComputedConstants,
    calendar_year: int,
    nut_stat: Float[np.ndarray, " ncols"],
    to_ba: float,
) -> tuple[State, Outputs, CuttingOutputs]
```

Extracted from current `cutting()` method. No mutation.
- Compute `cut_stems`, logging residues, harvested volumes
- Update `remaining_share`, reset `agearr` to 1 for cut columns
- Return `(new_state, apply_allometry(...), CuttingOutputs(...))`

---

## 3. `stand.py` — dataclasses

```python
@dataclass(frozen=True)
class Params:
    """Bundles the 3 canopy layer params. susi_main passes this, never touches canopylayer."""
    dominant: canopylayer.Params
    subdominant: canopylayer.Params
    under: canopylayer.Params

@dataclass(frozen=True)
class ComputedConstants:
    """Bundles the 3 canopy layer computed constants."""
    dominant: canopylayer.ComputedConstants
    subdominant: canopylayer.ComputedConstants
    under: canopylayer.ComputedConstants

@dataclass(frozen=True)
class State:
    """Stand state — the only vegetation state susi_main threads through the loop."""
    nut_stat: Float[np.ndarray, " ncols"]
    dominant: canopylayer.State
    subdominant: canopylayer.State
    under: canopylayer.State

@dataclass(frozen=True)
class Outputs:
    """Hectare-basis aggregates from all 3 layers. Same fields as canopylayer.Outputs
    but converted /tree → /ha via stems, plus stand-only fields."""

    # Identical fields to canopylayer.Outputs, but in /ha basis
    basalarea: Float[np.ndarray, " ncols"]
    biomass: Float[np.ndarray, " ncols"]
    hdom: Float[np.ndarray, " ncols"]
    leafarea: Float[np.ndarray, " ncols"]
    leafmass: Float[np.ndarray, " ncols"]
    stems: Float[np.ndarray, " ncols"]
    volume: Float[np.ndarray, " ncols"]
    volumegrowth: Float[np.ndarray, " ncols"]
    yi: Float[np.ndarray, " ncols"]
    logvolume: Float[np.ndarray, " ncols"]
    pulpvolume: Float[np.ndarray, " ncols"]

    # Stand-only
    mean_diameter: Float[np.ndarray, " ncols"]
    biomassgrowth: Float[np.ndarray, " ncols"]

    # ... same litter fields (finerootlitter, nonwoodylitter, woodylitter, all with N/P/K)
    # ... same mortality litter fields
    # ... same nutrient demand fields
    # ... same logging residue fields (populated from CuttingOutputs in cutting years)
    # ... same harvested volume fields

@dataclass(frozen=True)
class ForcingInputs:
    """Bundled annual inputs for stand.grow_stand() and stand.assimilate_stand()."""
    photopara: Any
    forc: pd.DataFrame = field(doc="annual weather")
    wt: pd.DataFrame = field(doc="annual water table")
    afp: pd.DataFrame = field(doc="annual air-filled porosity")
    n_supply: Float[np.ndarray, " ncols"]
    p_supply: Float[np.ndarray, " ncols"]
    k_supply: Float[np.ndarray, " ncols"]
    groundvegetation_outputs: Any
    previous_nut_stat: Float[np.ndarray, " ncols"]
    calendar_year: int
    cutting_to_ba: float | None = None
```

---

## 4. `stand.py` — functions

### `compute_constants`

```python
def compute_constants(
    params: Params,
    allometry_params,
) -> ComputedConstants
```

Calls `canopylayer.compute_constants()` 3 times internally with the per-layer data from `allometry_params`. Returns `ComputedConstants(dominant=..., subdominant=..., under=...)`.

### `compute_initial_state`

```python
def compute_initial_state(
    params: Params,
    cc: ComputedConstants,
    agearr: dict[str, np.ndarray],
    ncols: int,
) -> tuple[State, Outputs]
```

- `nut_stat = np.ones(ncols)`
- Call `canopylayer.compute_initial_state()` for each of the 3 layers
- `_aggregate()` → first year's stand outputs
- Return `(State(nut_stat, dom_state, sub_state, under_state), stand_out)`

### `_aggregate`

```python
def _aggregate(
    dom_out: canopylayer.Outputs,
    sub_out: canopylayer.Outputs,
    under_out: canopylayer.Outputs,
) -> Outputs
```

Pure. Replaces `Stand.update()` + `Stand.reset_vars()`.
- For each field: `stand_field = cl_out.field * cl_out.stems`, sum across 3 layers
- `hdom = np.maximum(...)`, `mean_diameter = weighted_avg(Dg, stems)`
- `biomassgrowth = biomass - biomass from previous aggregation`
Return `Outputs(...)`.

### `_compute_lai_above`

```python
def _compute_lai_above(
    dom_allom: canopylayer.Outputs,
    sub_allom: canopylayer.Outputs,
    under_allom: canopylayer.Outputs,
) -> tuple[Float[np.ndarray, " ncols"], Float[np.ndarray, " ncols"], Float[np.ndarray, " ncols"]]
```

Extracted from `Stand.assimilate()` lines 516-545. Stack hdom, argsort descending, order LAI by height, cumulative sum. Return `(lai_above_dom, lai_above_sub, lai_above_under)`.

### `grow_stand`

```python
def grow_stand(
    state: State,
    cc: ComputedConstants,
    inputs: ForcingInputs,
) -> tuple[State, Outputs]
```

**Bundles the 3 layers.** Body:

```
# 1. Apply allometry on each layer's current state (for lai_above + growth inputs)
dom_allom = canopylayer.apply_allometry(state.dominant.biomass, state.dominant.agearr, state.dominant.remaining_share, cc.dominant)
sub_allom = canopylayer.apply_allometry(...)
under_allom = canopylayer.apply_allometry(...)

# 2. Height-order for light competition
lai_dom, lai_sub, lai_under = _compute_lai_above(dom_allom, sub_allom, under_allom)

# 3. Grow each layer
dom_state, dom_out = canopylayer.grow_stand(
    state.dominant, cc.dominant,
    inputs.photopara, inputs.forc, inputs.wt, inputs.afp,
    inputs.previous_nut_stat, state.nut_stat, lai_dom,
)
sub_state, sub_out = canopylayer.grow_stand(...)
under_state, under_out = canopylayer.grow_stand(...)

# 4. Aggregate
stand_out = _aggregate(dom_out, sub_out, under_out)

# 5. Return
new_stand_state = State(state.nut_stat, dom_state, sub_state, under_state)
return new_stand_state, stand_out
```

### `update_nutrient_status`

```python
def update_nutrient_status(
    state: State,
    stand_out: Outputs,
    inputs: ForcingInputs,
) -> State
```

Pure. Extract from `Stand.update_nutrient_status()`.
- Compute supply/demand ratios with Reineke density modifier
- Apply delay ODE: `nut_stat = nut_stat + (ratio - nut_stat) / tau`
- Clip to `[0.5, 2.0]`
Return `replace(state, nut_stat=new_nut_stat)` — canopy layer states unchanged.

### `cut_stand`

```python
def cut_stand(
    state: State,
    cc: ComputedConstants,
    inputs: ForcingInputs,
) -> tuple[State, Outputs, canopylayer.CuttingOutputs | None]
```

- Only applies to dominant layer (matches current OOP behavior)
- Call `canopylayer.cut_stand(state.dominant, cc.dominant, inputs.calendar_year, state.nut_stat, inputs.cutting_to_ba)`
- Re-aggregate: `_aggregate(dom_out, sub_out, under_out)` (sub/under unchanged)
- Return `(state, stand_out, cutting_out)`

### `assimilate_stand`

```python
def assimilate_stand(
    state: State,
    cc: ComputedConstants,
    inputs: ForcingInputs,
) -> tuple[State, Outputs, canopylayer.CuttingOutputs | None]
```

**Full annual orchestrator.** Body:

```python
# 1. Growth
new_state, stand_out = grow_stand(state, cc, inputs)

# 2. Cutting (before nutrient update, matching OOP order)
cut_out = None
if inputs.cutting_to_ba is not None and inputs.cutting_to_ba < 1.0:
    new_state, stand_out, cut_out = cut_stand(new_state, cc, inputs)
    # cut_stand uses state.nut_stat (pre-update) internally

# 3. Nutrient status
new_state = update_nutrient_status(new_state, stand_out, inputs)

return new_state, stand_out, cut_out
```

---

## 5. Changes to `susi_main.py`

### `ModuleParams`

```python
@dataclass(frozen=True)
class ModuleParams:
    stand: stand.Params                       # NEW — bundles 3 canopylayer.Params
    canopygrid: canopygrid.Params
    mosslayer: mosslayer.Params
    methane: methane.Params
    temperature: temperature.Params
    fertilization: fertilization.Params
    strip: strip.Params
```

### `_build_params()` — construct StandParams

```python
sp = susi_params.site_parameters
n = sp.n
return ModuleParams(
    stand=stand.Params(
        dominant=canopylayer.Params(
            name="dominant",
            ncols=n,
            nlyrs=np.unique(sp.canopylayers.dominant),
            sfc=sp.sfc.copy(),
        ),
        subdominant=canopylayer.Params(name="subdominant", ...),
        under=canopylayer.Params(name="under", ...),
    ),
    ...
)
```

### Initialization — replace `Stand()` + `stand.update()`

```python
# Build stand-level constants (calls canopylayer.compute_constants ×3 internally)
stand_cc = stand.compute_constants(module_params.stand, allometry_params)

# Initialize (replaces Stand constructor + .update())
stand_state, stand_out = stand.compute_initial_state(
    module_params.stand, stand_cc, sp.age, ncols=n,
)
```

### Per-scenario reset

```python
stand_state, stand_out = stand.compute_initial_state(
    module_params.stand, stand_cc, sp.age, ncols=n,
)
```

### Annual loop — one call replaces assimilate + update + cutting + update_nutrient_status + reset_logging

```python
forcings = stand.ForcingInputs(
    photopara=self.parameters.photo_parameters,
    forc=self.weather_forcing.loc[str(calendar_year)],
    wt=dfwt.loc[str(calendar_year)],
    afp=dfafp.loc[str(calendar_year)],
    n_supply=yr_out_N.out_root_lyr + sp.depoN + fert_outputs.nutrient_release["N"],
    p_supply=yr_out_P.out_root_lyr + sp.depoP + fert_outputs.nutrient_release["P"],
    k_supply=yr_out_K.out_root_lyr + sp.depoK + fert_outputs.nutrient_release["K"],
    groundvegetation_outputs=gv_outputs,
    previous_nut_stat=stand_state.nut_stat,
    calendar_year=calendar_year,
    cutting_to_ba=sp.cutting_to_ba if calendar_year == sp.cutting_yr else None,
)

stand_state, stand_out, cutting_out = stand.assimilate_stand(
    stand_state, stand_cc, forcings,
)
```

### Attribute reads — old → new

| Old | New |
|---|---|
| `stand.volume` | `stand_out.volume` |
| `stand.stems` | `stand_out.stems` |
| `stand.basalarea` | `stand_out.basalarea` |
| `stand.leafarea` | `stand_out.leafarea` |
| `stand.hdom` | `stand_out.hdom` |
| `stand.nut_stat` | `stand_state.nut_stat` |
| `stand.nonwoodylitter` | `stand_out.nonwoodylitter` |
| `stand.woodylitter` | `stand_out.woodylitter` |
| `stand.n_demand` | `stand_out.n_demand` |
| `stand.nonwoody_lresid` | `cutting_out.nonwoody_lresid` (or stand_out in cut years) |
| `stand.dominant` | `stand_state.dominant` |
| `stand.subdominant` | `stand_state.subdominant` |
| `stand.under` | `stand_state.under` |

Stand-alone attribute reads in susi_main that remain after the annual call:
- `stand.volume`, `.stems`, `.basalarea` → used for gvegetation assembly (before assimilate, after previous year's growth) → `stand_out` from the previous year
- `stand.nut_stat` → used for `canopygrid.update_amax()` → `stand_state.nut_stat`
- `stand.hdom`, `.leafarea` → used in daily canopygrid loop → `stand_out.hdom`, `.leafarea` from previous year

`stand.reset_logging()` is removed — each year's `stand_out` is fresh, with logging residue fields zero unless cutting fired.

---

## 6. Implementation order (TDD — red/green per step)


| # | File | Description |
|---|------|-------------|
| 1a | `canopylayer.py` | Define dataclasses: `Params`, `ComputedConstants`, `State`, `Outputs`, `CuttingOutputs` |
| 1b | `canopylayer.py` | Extract `_leaf_dynamics` as module-level private function |
| 2 | `canopylayer.py` | `apply_allometry()` — maps core state `(biomass, agearr, remaining_share)` → all derived allometric outputs |
| 3 | `canopylayer.py` | `compute_constants()` — per-zone interpolation functions via `build_allometry_interpolation_functions()`, builds `tree_species` |
| 4 | `canopylayer.py` | `compute_initial_state()` — initialises `State(agearr, biomass, remaining_share=1.0)` from age-based bm interpolation |
| 5 | `canopylayer.py` | `grow_stand()` — applies allometry, calls `assimilation_yr()`, per-zone `_leaf_dynamics()`, biomass increment, re-apply allometry, merge outputs |
| 6 | `canopylayer.py` | `cut_stand()` — thinning (keep `to_ba` fraction of BA) or clear-cut, computes logging residues + harvested volumes, updates `remaining_share`, resets age |
| 7 | `stand.py` | Define dataclasses: `Params`, `ComputedConstants`, `State`, `Outputs`, `ForcingInputs` |
| 8 | `stand.py` | `_aggregate()` — sums 3 canopylayer outputs to per-ha basis (per-tree fields × stems, cutting/ha fields summed directly) |
| 9 | `stand.py` | `_compute_lai_above()` — height-order LAI for correct shading between layers |
| 10 | `stand.py` | `compute_constants()` — accepts per-column zone ID arrays, builds `ixs` dict, delegates to canopylayer ×3 |
| 11 | `stand.py` | `compute_initial_state()` — delegates to canopylayer ×3, aggregates |
| 12 | `stand.py` | `grow_stand()` — delegates to canopylayer ×3 with LAI_above, aggregates |
| 13 | `stand.py` | `update_nutrient_status()` — N/P/K supply vs demand ratio with delay ODE (τ=3), clips to [0.5, 2.0] |
| 14 | `stand.py` | `cut_stand()` — applies cutting only to dominant layer (matches existing OOP behaviour) |
| — | `stand.py` | `assimilate_stand()` — full annual orchestrator: grow → cut (if cutting year) → nutrient update |
| — | Both | Lint (`ruff`) and type-check (`ty`) pass clean on both `canopylayer.py` and `stand.py` |
| 15 | `susi_main.py` | Add `stand.Params` to `ModuleParams` + update `_build_params()` |
| 16 | `susi_main.py` | Replace all OOP Stand/Canopylayer usage with functional calls through `stand.py` |
| 17 | — | Golden file test: `pytest tests/golden_file_test/` passes |
| 18 | — | Clean up old OOP classes and imports |

---


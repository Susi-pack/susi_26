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
    """Zero-filled in non-cutting years (no Optional union — JAX-friendly)."""
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

@dataclass(frozen=True)
class LeafDynamicsOutputs:
    """Named return from _leaf_dynamics (replaces anonymous tuple)."""
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

@dataclass(frozen=True)
class Inputs:
    """Annual inputs for canopylayer.grow_stand()."""
    photopara: Any
    forc: pd.DataFrame = field(doc="annual weather")
    wt: pd.DataFrame = field(doc="annual water table")
    afp: pd.DataFrame = field(doc="annual air-filled porosity")
    previous_nut_stat: Float[np.ndarray, " ncols"]
    nut_stat: Float[np.ndarray, " ncols"]
    lai_above: Float[np.ndarray, " ncols"]
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
    inputs: Inputs,
) -> tuple[State, Outputs]
```

The annual growth computation. Body:

```python
# 1. Apply allometry on current state (gives stems, leafarea, finerootlitter, woodylitter)
allom = apply_allometry(state.biomass, state.agearr, state.remaining_share, cc)

# 2. Photosynthesis
LAI_photo = allom.leafarea * 2 * allom.stems          # double-sided
npp, npp_pot = assimilation_yr(
    inputs.photopara, inputs.forc, inputs.wt, inputs.afp,
    LAI_photo, inputs.lai_above * 2,
)

# Convert to tree basis (handle zero stems)
mask = allom.stems > 0
npp = np.divide(npp * inputs.nut_stat, allom.stems, out=np.full_like(npp, np.nan), where=mask)
npp_pot = np.divide(npp_pot * inputs.nut_stat, allom.stems, out=np.full_like(npp_pot, np.nan), where=mask)

# 3. Leaf dynamics
ld = _leaf_dynamics(
    state.biomass, npp, allom.leafmass,
    inputs.previous_nut_stat, inputs.nut_stat, state.agearr,
    cc.allodic, cc.tree_species,
)

# 4. Biomass increment (leaf dynamics allocates carbon to leaves; rest goes to stem)
delta_bm = npp - ld.C_consumption - allom.finerootlitter - allom.woodylitter
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
    new_lmass=ld.new_lmass,
    leaf_litter=ld.leaf_litter,
    C_consumption=ld.C_consumption,
    Nleafdemand=ld.Nleafdemand, Nleaf_litter=ld.Nleaf_litter, N_leaf=ld.N_leaf,
    Pleafdemand=ld.Pleafdemand, Pleaf_litter=ld.Pleaf_litter, P_leaf=ld.P_leaf,
    Kleafdemand=ld.Kleafdemand, Kleaf_litter=ld.Kleaf_litter, K_leaf=ld.K_leaf,
    nonwoodylitter=new_allom.finerootlitter + ld.leaf_litter,
    n_nonwoodylitter=new_allom.n_finerootlitter + ld.Nleaf_litter,
    p_nonwoodylitter=new_allom.p_finerootlitter + ld.Pleaf_litter,
    k_nonwoodylitter=new_allom.k_finerootlitter + ld.Kleaf_litter,
)
return new_state, outputs
```

### `_leaf_dynamics`

```python
def _leaf_dynamics(
    bm, bm_increment, current_leafmass, previous_nut_stat, nut_stat,
    agenow, allometry_funcs, species, printOpt=False
) -> LeafDynamicsOutputs
```

Already structurally pure — extract as module-level private function returning `LeafDynamicsOutputs`. No changes to the computation.

### `compute_constants`

```python
def compute_constants(
    params: Params,
    allometry_data: dict[int, pd.DataFrame],
    species_id: dict[int, int],
) -> ComputedConstants
```

For each non-zero zone in `params.nlyrs`:
- Call `build_allometry_interpolation_functions(AllometryParams(species=species_id[z], site_fertility_class=sfc_median), allometry_data[z])`
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

Always returns a `CuttingOutputs` (zero-filled when no cutting occurs — no `Optional`).
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
class Inputs:
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
- `_aggregate(..., previous_stand_biomass=None)` → first year's stand outputs
- Return `(State(nut_stat, dom_state, sub_state, under_state), stand_out)`

### `_aggregate`

```python
def _aggregate(
    dom_out: canopylayer.Outputs,
    sub_out: canopylayer.Outputs,
    under_out: canopylayer.Outputs,
    dom_biomass: Float[np.ndarray, " ncols"],
    sub_biomass: Float[np.ndarray, " ncols"],
    under_biomass: Float[np.ndarray, " ncols"],
    previous_stand_biomass: Float[np.ndarray, " ncols"] | None = None,
) -> Outputs
```

Pure. Replaces `Stand.update()` + `Stand.reset_vars()`.
- For each per-tree field: `stand_field = cl_out.field * cl_out.stems`, sum across 3 layers
- `stems` summed directly, `hdom = np.maximum(...)`, `mean_diameter = weighted_avg(Dg, stems)`
- `biomass = sum(state_biomass × stems)` from layer state arrays (not in `canopylayer.Outputs`)
- `biomassgrowth = biomass - previous_stand_biomass` (zero when `previous_stand_biomass is None`)
- Cutting fields zeroed (filled later via `_merge_cutting_outputs`)
Return `Outputs(...)`.

### `_merge_cutting_outputs`

```python
def _merge_cutting_outputs(
    stand_out: Outputs,
    dom_cut: canopylayer.CuttingOutputs | None = None,
    sub_cut: canopylayer.CuttingOutputs | None = None,
    under_cut: canopylayer.CuttingOutputs | None = None,
) -> Outputs
```

Pure. Merges per-layer `CuttingOutputs` into a `stand.Outputs` (which has zeroed cutting fields from `_aggregate`).
For each cutting field: sum across layers (values are already per-ha from `canopylayer.cut_stand`).
Returns `dataclasses.replace(stand_out, harvested_volume=..., ...)`.
Each `cut` arg is optional (None = no cutting in that layer = zeros).

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
    inputs: Inputs,
) -> tuple[State, Outputs]
```

**Bundles the 3 layers.** Body:

```python
# 1. Apply allometry on each layer's current state (for lai_above + growth inputs)
dom_allom = canopylayer.apply_allometry(state.dominant.biomass, state.dominant.agearr, state.dominant.remaining_share, cc.dominant)
sub_allom = canopylayer.apply_allometry(...)
under_allom = canopylayer.apply_allometry(...)

# 2. Height-order for light competition
lai_dom, lai_sub, lai_under = _compute_lai_above(dom_allom, sub_allom, under_allom)

# 3. Build per-layer Inputs with correct lai_above
dom_inputs = canopylayer.Inputs(
    photopara=inputs.photopara, forc=inputs.forc, wt=inputs.wt, afp=inputs.afp,
    previous_nut_stat=inputs.previous_nut_stat, nut_stat=state.nut_stat,
    lai_above=lai_dom,
)
sub_inputs = dataclasses.replace(dom_inputs, lai_above=lai_sub)
under_inputs = dataclasses.replace(dom_inputs, lai_above=lai_under)

# 4. Grow each layer
dom_state, dom_out = canopylayer.grow_stand(state.dominant, cc.dominant, dom_inputs)
sub_state, sub_out = canopylayer.grow_stand(state.subdominant, cc.subdominant, sub_inputs)
under_state, under_out = canopylayer.grow_stand(state.under, cc.under, under_inputs)

# 5. Compute previous biomass for growth calculation
prev_biomass = (
    state.dominant.biomass * dom_allom.stems
    + state.subdominant.biomass * sub_allom.stems
    + state.under.biomass * under_allom.stems
)

# 6. Aggregate (dom_out, sub_out, under_out are the outputs from grow_stand;
#    layer state biomass is passed separately — not in canopylayer.Outputs)
stand_out = _aggregate(
    dom_out, sub_out, under_out,
    dom_state.biomass, sub_state.biomass, under_state.biomass,
    previous_stand_biomass=prev_biomass,
)

# 7. Return
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
    inputs: Inputs,
) -> tuple[State, Outputs, canopylayer.CuttingOutputs]
```

Always returns a `CuttingOutputs` (zero-filled in non-cutting years — no `Optional`).
- Only applies to dominant layer (matches current OOP behavior)
- Call `canopylayer.cut_stand(state.dominant, cc.dominant, dom_out, state.nut_stat, inputs.cutting_to_ba)` → `(dom_state, dom_cut)`
- Re-aggregate: `_aggregate(dom_state, sub_state, under_state, dom_out, sub_out, under_out)` with updated state biomass
- Merge cutting outputs: `_merge_cutting_outputs(stand_out, dom_cut=dom_cut)`
- Return `(new_state, stand_out, cutting_out)` where `cutting_out` is the dominant `CuttingOutputs`

### `assimilate_stand`

```python
def assimilate_stand(
    state: State,
    cc: ComputedConstants,
    inputs: Inputs,
) -> tuple[State, Outputs, canopylayer.CuttingOutputs]
```

**Full annual orchestrator.** Body:

```python
# 1. Growth
new_state, stand_out = grow_stand(state, cc, inputs)

# 2. Cutting (before nutrient update, matching OOP order)
if inputs.cutting_to_ba is not None and inputs.cutting_to_ba < 1.0:
    new_state, stand_out, cut_out = cut_stand(new_state, cc, inputs)
    # cut_stand internally calls _merge_cutting_outputs to fill cutting fields
else:
    cut_out = canopylayer.CuttingOutputs(
        *[np.zeros_like(state.nut_stat) for _ in fields(canopylayer.CuttingOutputs)]
    )
    stand_out = _merge_cutting_outputs(stand_out)

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
stand_inputs = stand.Inputs(
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
    stand_state, stand_cc, stand_inputs,
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
| `stand.nonwoody_lresid` | `cutting_out.nonwoody_lresid` (always valid; zero in non-cutting years) |
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

Each step: **write failing test → make pass → `ruff` / `ty` clean → commit → append progress to this document**.

| # | File | Description |
|---|------|-------------|
| 1a | `canopylayer.py` | Define dataclasses: `Params`, `ComputedConstants`, `State`, `Outputs`, `CuttingOutputs`, `LeafDynamicsOutputs`, `Inputs` |
| 1b | `canopylayer.py` | Extract `_leaf_dynamics` as module-level private function returning `LeafDynamicsOutputs` |
| 2 | `canopylayer.py` | `apply_allometry()` — maps core state `(biomass, agearr, remaining_share)` → all derived allometric outputs |
| 3 | `canopylayer.py` | `compute_constants()` — per-zone interpolation functions via `build_allometry_interpolation_functions()`, builds `tree_species` |
| 4 | `canopylayer.py` | `compute_initial_state()` — initialises `State(agearr, biomass, remaining_share=1.0)` from age-based bm interpolation |
| 5 | `canopylayer.py` | `grow_stand()` — takes `state, cc, inputs: Inputs`, calls `assimilation_yr()` + `_leaf_dynamics()`, biomass increment, re-apply allometry, merge outputs |
| 6 | `canopylayer.py` | `cut_stand()` — thinning (keep `to_ba` fraction of BA) or clear-cut, computes logging residues + harvested volumes, updates `remaining_share`, resets age |
| 7 | `stand.py` | Define dataclasses: `Params`, `ComputedConstants`, `State`, `Outputs`, `Inputs` |
| 8 | `stand.py` | `_aggregate()` — sums 3 canopylayer outputs to per-ha basis; takes layer biomass arrays + `previous_stand_biomass` for growth calc |
| — | `stand.py` | `_merge_cutting_outputs()` — merges per-layer `CuttingOutputs` into `stand.Outputs` (fills harvested/residue fields) |
| 9 | `stand.py` | `_compute_lai_above()` — height-order LAI for correct shading between layers |
| 10 | `stand.py` | `compute_constants()` — accepts per-column zone ID arrays, builds `ixs` dict, delegates to canopylayer ×3 |
| 11 | `stand.py` | `compute_initial_state()` — delegates to canopylayer ×3, aggregates with `previous_stand_biomass=None` |
| 12 | `stand.py` | `grow_stand()` — builds per-layer `canopylayer.Inputs`, delegates to canopylayer ×3 with LAI_above, aggregates |
| 13 | `stand.py` | `update_nutrient_status()` — N/P/K supply vs demand ratio with delay ODE (τ=3), clips to [0.5, 2.0] |
| 14 | `stand.py` | `cut_stand()` — applies cutting to dominant layer; re-aggregates + merges cutting fields; always returns `CuttingOutputs` |
| — | `stand.py` | `assimilate_stand()` — full annual orchestrator: grow → cut (or zero `CuttingOutputs`) → nutrient update |
| — | Both | Lint (`ruff`) and type-check (`ty`) pass clean on both `canopylayer.py` and `stand.py` |
| 15 | `susi_main.py` | Add `stand.Params` to `ModuleParams` + update `_build_params()` |
| 16 | `susi_main.py` | Replace all OOP Stand/Canopylayer usage with functional calls through `stand.py`; update `stand.ForcingInputs` → `stand.Inputs` |
| 17 | — | Golden file test: `pytest tests/golden_file_test/` passes |
| 18 | — | Clean up old OOP classes and imports |

---

## 7. Intentional Divergences from SuperSUSI_functional_architecture.md

### `canopylayer` is not a first-class module

`canopylayer` lacks its own `assemble_inputs()` / `run_timestep()` protocol entry points. It is deliberately hidden behind `stand.py` — `susi_main.py` never imports `canopylayer` directly. This is a pragmatic choice to keep the orchestrator lean: stacking 3 canopy layers per stand into the orchestration loop would triple the coupling calls without scientific benefit.

### No `run_timestep()` naming

The architecture doc suggests `run_timestep()` as the standard module entry point. Instead we use `grow_stand()` / `assimilate_stand()` at stand level and `grow_stand()` at canopy level. These names match the domain language and distinguish growth from the full assimilation cycle that includes cutting and nutrient update.

### `Inputs` bundles module-owned and external data

The architecture prescribes `Inputs` as a bundle of *other* categories. Here `canopylayer.Inputs` includes `nut_stat` (owned by stand) alongside `forc`, `wt`, `afp` (owned by weather). This is simpler than requiring susi_main to assemble separate per-layer inputs per year.

### `_aggregate` receives `previous_stand_biomass` explicitly

The architecture says outputs should not feed back into state — but `biomassgrowth` *is* an output that depends on the previous year's biomass. Making the previous year's biomass an explicit parameter keeps the function pure.

### `_aggregate` also needs layer biomass arrays

`canopylayer.Outputs` does not contain a `biomass` field (biomass is part of `canopylayer.State`, not the allometric outputs). So `_aggregate` accepts `dom_biomass`, `sub_biomass`, `under_biomass` separately to compute the stand-level `biomass` field.

### `_merge_cutting_outputs` is separate from `_aggregate`

The architecture prescribes one aggregation function. Cutting fields (harvested_*, *_lresid) are zeroed by `_aggregate` and filled later by `_merge_cutting_outputs` when cutting occurs. This keeps `_aggregate` focused on growth aggregation and avoids branching on whether cutting happened.

### `canopylayer.compute_constants` uses per-zone data dicts

The plan initially proposed `(params, allometry_df: pd.DataFrame, species_id: int)` but the multiple canopy layer zones require per-zone data, hence `allometry_data: dict[int, pd.DataFrame]` and `species_id: dict[int, int]`.

### `canopylayer.cut_stand` receives `out: Outputs` not `calendar_year`

The plan initially proposed `cut_stand(state, cc, calendar_year, nut_stat, to_ba)`. The actual signature is `cut_stand(state, cc, out, nut_stat, to_ba)` — it needs the allometry output to read basalarea, stems, and leaf mass for computing cut amounts. `calendar_year` was unused.

---

## 9. Root Cause: Golden File Discrepancy

### Background

The golden file test compares the functional `supersusi` against a golden `netCDF` file produced by the OOP `susi`. The first failure is at year 0, `/stand/basalarea` = **16.07** (golden) vs **16.52** (functional).

### Mechanism

**OOP initialization** (`Canopylayer.__init__` → `initialize_domain()`) uses a **mixed** formula assignment:

| Field | OOP init formula | OOP update formula (after `Canopylayer.update()`) |
|-------|-----------------|---------------------------------------------------|
| `biomass` | `age_based.bm(age)` | (`bm` from caller) |
| `basalarea` | `age_based.ba(age)` | `bmToBa(bm)` |
| `hdom` | `age_based.hdom(age)` | `bmToHdom(bm)` |
| `Dg` | `bmToDg(bm)` | `bmToDg(bm)` |
| `leafarea` | `bmToLai(bm) × nut_stat` | `bmToLai(bm)` |
| `leafmass` | `ageToLeaves(age) × nut_stat` | `bmToLeafMass(bm)` |
| `volume` | `bmToVol(bm)` | `bmToVol(bm)` |
| `stems` | `bmToStems(bm) × remaining_share` | `bmToStems(bm) × remaining_share` |

After `initialize_domain()`, the OOP **does NOT call `Canopylayer.update()`** on the layers. `Stand.update()` only aggregates layer values to ha-basis. So the OOP's initial (year 0) values use the mixed formulas from `initialize_domain()`.

**Functional `compute_initial_state()`** (`canopylayer.py:511`) computes `biomass` from `age_based.bm(age)` then feeds it through `apply_allometry()`, which uses **biomass-based formulas for ALL fields**:

| Field | Functional formula |
|-------|-------------------|
| `basalarea` | `bmToBa(bm)` ← **different from OOP init** |
| `hdom` | `bmToHdom(bm)` ← **different from OOP init** |

For basalarea specifically:
- OOP init: `age_based.ba(age=60)` = **0.01846 m²/tree**
- Functional: `bmToBa(bm=83.95)` = **0.01898 m²/tree**

These differ because `ageToBa` and `bmToBa` are distinct interpolation functions in the allometric dataset. At age 60 / biomass 83.95, they produce values that differ by ~2.8%.

### Why this matters

The golden file was generated from the OOP, so year-0 values match the mixed `initialize_domain()` formulas. The functional version produces biomass-based values which differ. **After year 1**, both versions use biomass-based formulas (the OOP's annual `Canopylayer.update()` is biomass-only), so the discrepancy only affects year 0.

### Fix required

**Option A**: Rewrite `canopylayer.compute_initial_state()` to reproduce the exact mixed formula assignment from `initialize_domain()`. This means writing an `initialize_domain()` (separate from `apply_allometry`) that uses:
- `age_based.bm(age)` for biomass
- `age_based.ba(age)` for basalarea
- `age_based.hdom(age)` for hdom
- `bmToDg(biomass)` for Dg
- `bmToLai(biomass) × nut_stat` for leafarea
- `age_based.leaves(age) × nut_stat` for leafmass
- `bmToStems(biomass) × 1.0` for stems
- `bmToVol(biomass)` for volume

(Note: `nut_stat` is always 1.0 at initialization, making the leafarea/leafmass scaling a no-op, but it must be present for API correctness.)

**Option B**: Regenerate the golden file to match the functional version's biomass-based initialization, then accept that year 0 differs from the OOP. This is simpler but means year-0 data is inconsistent with the reference.

### Constraint violation: `canopylayer` leaked into `susi_main.py`

The design plan (§1 Key Decisions) states `stand.py` is the only module `susi_main.py` imports for vegetation. However, `susi_main.py` now imports `canopylayer` for:
1. `_build_params()` — constructing `canopylayer.Params` instances (called 3×, once per layer)
2. Annual `canopylayer.apply_allometry()` calls — used to produce `write_canopy_layer` output data from per-layer state/constants

Fix for Phase 6: Move the `write_canopy_layer` allometry computation into `stand_mod` (expose a function that returns per-layer allometry outputs for given stand state), or make `write_canopy_layer` accept per-layer state+constants and call `apply_allometry` internally (which still requires canopylayer... unless `outputs.py` already imports it). Actually `outputs.py` already imports `canopylayer`. So the leak is acceptable for now if we accept the design constraint relaxation.

---

## 8. Progress Log

| Date | Step | What was done | Tests | Lint/Type |
|------|------|---------------|-------|-----------|
| 2026-06-05 | 1a | Add frozen dataclasses to `canopylayer.py`: `Params`, `ComputedConstants`, `State`, `Outputs`, `CuttingOutputs`, `LeafDynamicsOutputs`, `Inputs` | 9 new tests pass; full suite 155/155 | ruff ✅, ty ✅ |
| 2026-06-05 | 1b | Extract `_leaf_dynamics()` as module-level function returning `LeafDynamicsOutputs`; `Canopylayer.leaf_dynamics()` becomes compat wrapper | 155/155 | ruff ✅, ty ✅ |
| 2026-06-05 | 2 | `apply_allometry()` — maps core state `(biomass, agearr, remaining_share, cc)` → all derived allometric outputs; growth fields zeroed; `nonwoodylitter = finerootlitter` | 158/158 (3 new tests) | ruff ✅, ty ✅ |
| 2026-06-05 | 3 | `compute_constants()` — per-zone AllometryFunctions via `build_allometry_interpolation_functions()`, builds `tree_species` array | 15 tests (1 new: minimal smoke test with 3-row dataframe) | ruff ✅, ty ✅ |
| 2026-06-05 | 4 | `compute_initial_state()` — derives `State(agearr, biomass, remaining_share=1.0)` from age via zone-specific `age_based.bm()`; returns `(State, Outputs)` tuple | 15 tests (1 new: shape/value invariants) | ruff ✅, ty ✅ |
| 2026-06-05 | 5 | `grow_stand()` — annual growth: `assimilation_yr` → `_leaf_dynamics` → biomass increment → `apply_allometry` at new biomass → merge growth fields, `volumegrowth`, `nonwoodylitter = finerootlitter + leaf_litter` | 16 tests (1 new: smoke test with mock allometry, 10 cols, 3-day weather) | ruff ✅, ty ✅ |
| 2026-06-05 | 6 | `cut_stand()` — thinning (`to_ba ≥ 1`: retain to_ba m²/ha BA, update `remaining_share`) or clear-cut (`to_ba < 1`: age=1, re-init from allometry); per-hectare residues + harvested volumes with wood-density conversion | 17 tests (1 new: thinning smoke test) | ruff ✅, ty ✅ |
| 2026-06-05 | 7 | `stand.py` — frozen dataclasses: `Params` (3× canopylayer.Params), `ComputedConstants` (3× canopylayer.ComputedConstants), `State` (nut_stat + 3× canopylayer.State), `Outputs` (all fields per-ha + stand-only + cutting), `Inputs` (bundled annual inputs) | 7 tests (new `tests/core/test_stand.py`) | ruff ✅, ty ✅ |
| 2026-06-05 | 8 | `_aggregate()` — pure function summing 3 canopylayer.Outputs to per-ha stand.Outputs; per-tree fields × stems, hdom=max, Dg weighted by stems for mean_diameter, biomass from layer states × stems; accepts `previous_stand_biomass` for biomassgrowth; safe division for zero stems; cutting fields zeroed | 10 tests (3 new: per-field sum, None previous, zero stems) | ruff ✅, ty ✅ |
| 2026-06-05 | — | Update plan signatures to match actual implementation; add `_merge_cutting_outputs` to plan as needed helper | — | — |
| 2026-06-05 | 9 | `_compute_lai_above()` — stack hdom ×3, argsort descending, reorder LAI (`leafarea * stems`) by height, cumsum, remap back to layer order; returns `(lai_above_dom, lai_above_sub, lai_above_under)` | 2 tests (shape, expected values for ordered heights) | ruff ✅, pytest 94/94 ✅ |
| 2026-06-05 | 13 | `stand.update_nutrient_status()` — N/P/K supply/(demand + leaf_demand + GV demand) × Reineke density modifier, delay ODE (τ=3), clip to [0.5, 2.0] | 2 tests (clip bounds, drift toward ratio) | ruff ✅, pytest 102/102 ✅ |
| 2026-06-05 | 14 | `stand.cut_stand()` — delegate to canopylayer, re-aggregate with updated biomass, merge cutting outputs | 2 tests (type check, to_ba=1.0 no-op) | ruff ✅, pytest 104/104 ✅ |
| 2026-06-05 | — | `stand.assimilate_stand()` — full annual orchestrator: grow → cut/zero → nutrient update | 2 tests (type/shape, cutting year) | ruff ✅, pytest 106/106 ✅ |
| 2026-06-05 | 15–16 | **Phase 2**: Added `stand: stand_mod.Params` to `ModuleParams`; constructed it in `_build_params()` from `susi_params.site_parameters.canopylayers`. **Phase 3**: Replaced all OOP Stand/Canopylayer usage in `susi_main.py` with functional calls: `compute_constants()` + `compute_initial_state()` → `grow_stand()` → `cut_stand()` → `update_nutrient_status()`; removed `Stand` import, `stand.reset_logging()`. **Phase 4**: Updated `outputs.py` `write_stand()`, `write_canopy_layer()`, `write_carbon_balance()` for functional data structures. **Constraint violation**: `canopylayer` now imported in `susi_main.py` for `_build_params()` and `apply_allometry()`. | 106 unit tests ✅ | ruff ✅, ty ✅ |
| 2026-06-05 | 17 | **Golden file test FAILS.** Initial `/stand/basalarea` = 16.52 (functional) vs 16.07 (golden), Δ ≈ 2.8%. Root cause identified (see §9). **Not fixed yet.** | FAILS | — |


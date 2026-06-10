# SuperSUSI scan architecture

A design for a functional, JAX-ready `susi_main.py` orchestrator.

---

## Background
The simulation has two timestep loops: daily and yearly.
The daily simulations are performed at the start of every year in the yearly loop.

## Core principles
Dataclasses are pure structs: they must have zero methods.

Architecture: two nested `jax.lax.scan` over (year, day).
All accumulation is `scan`'s return value.

```
run(params, forcings)
  ├── _build_params()           # already functional
  ├── _compute_constants()      # already functional
  ├── _init_state()             # new: builds AllState from params + constants
  ├── jax.lax.scan(_run_annual_step, init_state, annual_forcings)
  │     ├── inner scan: _run_daily_step (hydrology + temperature)
  │     └── annual biogeochemistry (stand, esom, fertilization, methane)
  └── _write_outputs()          # imperative shell, outside scan
```
---

## Decisions

- **NumericalBuffer**: Deferred. `strip.NumericalBuffer` keeps in-place mutation. `_run_daily_step` is impure because of this. A future step will replace the strip solver.
- **dwt bug**: Not fixed. The current code never feeds `strip_state.H - constants.ele` back into `rew_drylimit`. The functional rewrite preserves this legacy behavior. Mark it with `# BUG: dwt frozen at initial value`.
- **Scenario loop**: Eliminated. Single scenario execution. Output arrays keep a leading `(1, ...)` dimension for backward compatibility with existing IO.
- **pandas DataFrames**: Deferred. JAX cannot handle them. They must be eliminated from all traced code paths before JAX conversion. IO code (imperative shell) may still use pandas. See Known Issues #5.

---

## Orchestrator dataclasses

State and Output is separated with the same logic of SuperSUSI_functional_architecture.md:
State is the `carry` in `jax.lax.scan`, Outputs are the variables not used in the next scan loop.

Because we have 2 `scan` loops (daily and annual), we need:

- `DailyState` (inner scan carry)
- `AnnualState` (outer scan carry, read-only during inner scan)
- `AllState` (top-level carry, bundles both)
- `DailyForcing`
- `DailyOutputs`
- `AnnualForcing`
- `AnnualOutputs`
- `SimulationOutput`


```python
@dataclass(frozen=True)
class DailyState:
    """Evolving inside the inner scan — updated every day."""
    canopy: canopygrid.State
    moss: mosslayer.State
    strip: strip.State
    peat_T: temperature.State


@dataclass(frozen=True)
class AnnualState:
    """Evolving inside the outer scan — updated once per year.
    Read-only during the inner daily scan.
    """
    stand: stand.State
    # stand_outputs[N] produced by year N's biogeochemistry,
    # consumed by year N+1's daily hydrology (hdom, leafarea).
    # Acts as de facto state because of the one-year delay.
    stand_outputs: stand.Outputs
    gv: gvegetation.State
    esom_mass: esom.State
    esom_N: esom.State
    esom_P: esom.State
    esom_K: esom.State


@dataclass(frozen=True)
class AllState:
    """Top-level carry for the outer scan."""
    daily: DailyState
    annual: AnnualState


@dataclass(frozen=True)
class DailyForcing:
    """One day's time-varying inputs."""
    T: float
    Prec: float
    Rg: float
    Par: float
    VPD: float
    h0ts_west: float
    h0ts_east: float


@dataclass(frozen=True)
class DailyOutputs:
    """Per-day outputs — inner scan's ys pytree."""
    wtd: np.ndarray                  # (n,) — strip_state.H - constants.ele
    afp: np.ndarray                  # (n,) — from strip
    T_soil_hydro: np.ndarray         # (n_hydro,) — from temperature
    delta: np.ndarray                # (n,)  potinf - transpi, coupling
    total_runoff: float              # ts_out.roff + mean(surface_runoff)
    surface_runoff: np.ndarray       # (n,) — from moss_returnflow
    interc: np.ndarray               # (n,) — from canopy
    evap: np.ndarray                 # (n,) — from canopy
    et: np.ndarray                   # (n,) — from canopy
    transpi: np.ndarray              # (n,) — from canopy
    efloor: np.ndarray               # (n,) — from canopy
    swe: np.ndarray                  # (n,) — from canopy


@dataclass(frozen=True)
class AnnualForcing:
    """All inputs for one year — outer scan iterates over these.
    The daily arrays (daily_T, daily_Rg, daily_VPD) are consumed
    by the annual biogeochemistry (stand assimilation, esom).
    They are the same data that the inner scan steps through
    day by day, replicated here for the annual post-processing
    (avoids re-extracting from stacked daily outputs).
    """
    # Daily arrays (used by annual biogeochemistry)
    daily_T: np.ndarray              # (MAX_DAYS,) air temperature
    daily_Rg: np.ndarray             # (MAX_DAYS,) global radiation — for assimilation_yr
    daily_VPD: np.ndarray            # (MAX_DAYS,) VPD — for assimilation_yr

    # Boundary conditions for the inner daily loop
    daily_Prec: np.ndarray           # (MAX_DAYS,) precipitation
    daily_Par: np.ndarray            # (MAX_DAYS,) PAR
    h0ts_west: np.ndarray            # (MAX_DAYS,) west ditch depth per day
    h0ts_east: np.ndarray            # (MAX_DAYS,) east ditch depth per day

    valid_days: int                  # 365 or 366
    calendar_year: int               # for phenology, management
    temp_sum: np.ndarray             # (n,) temperature sum for ground vegetation

    # Annual management flags
    do_cutting: bool
    cutting_to_ba: float


@dataclass(frozen=True)
class AnnualOutputs:
    """Per-year outputs — includes stacked daily outputs for IO + biogeochemistry."""
    daily: DailyOutputs              # stacked → (MAX_DAYS, *)
    stand: stand.Outputs
    stand_dom: canopylayer.Outputs
    stand_sub: canopylayer.Outputs
    stand_under: canopylayer.Outputs
    gv: gvegetation.Outputs
    esom_mass: esom.YearOutputs
    esom_N: esom.YearOutputs
    esom_P: esom.YearOutputs
    esom_K: esom.YearOutputs
    methane: methane.Outputs
    fertilization: fertilization.Outputs
    Rhet: np.ndarray                 # (n,)
    soil_co2_balance: np.ndarray     # (n,)
    doc_export: esom.DOCExportOutputs
    strip_diag: strip.ResidenceTimeOutput


```

---

## Implementation plan (TDD)

Each step has: **test first**, then **implementation**. Module rewrites (Steps 1-3) can be done in parallel.

### Steps 1-3: DEFERRED — DataFrame → array rewrites

These steps replace `pd.DataFrame` with numpy arrays in module interfaces. They are deferred to a later phase because:
- The functional rewrite of `susi_main.py` can proceed with DataFrames as-is (IO code already uses them)
- The DataFrame elimination is a prerequisite for JAX conversion, not for the functional architecture
- Keeping DataFrames now avoids touching module internals prematurely

**Deferred steps:**
1. `heterotrophic_respiration_yr`, `ojanen_2019` in `susi_utils.py`
2. `stand.Inputs` in `stand.py`
3. `methane.assemble_inputs` in `methane.py`

These will be tackled in a separate phase before JAX conversion. The `_run_annual_step` pseudocode below assumes DataFrames are still used (passing arrays would require module changes).

### Step 1: Define top-level dataclasses

**Test** (`tests/test_susi_main_types.py`):
- All dataclasses are frozen
- All dataclasses can be constructed with sample data
- `replace()` works on `AllState`, `DailyState`, `AnnualState`

**Implementation** (`src/supersusi/core/susi_main.py`):
Define `DailyState`, `AnnualState`, `AllState`, `DailyForcing`, `DailyOutputs`, `AnnualForcing`, `AnnualOutputs`, `SimulationOutput` as shown in the dataclass section above.

### Step 2: Write `_init_state()`

**Test** (`tests/test_susi_main_init.py`):
- `_init_state` returns an `AllState` with all fields populated
- Each module state matches calling its `compute_initial_state` directly
- `canopygrid.update_amax` is called (verify `canopy.amax` differs from raw initial when `nut_stat != 1`)

**Implementation** (`src/supersusi/core/susi_main.py`):
```python
def _init_state(
    params: ModuleParams,
    constants: ModuleComputedConstants,
    susi_params: SusiParams,
) -> AllState:
    stand_state, stand_out, dom_out, sub_out, under_out = stand.compute_initial_state(
        params.stand, constants.stand, susi_params.site_parameters.age, susi_params.site_parameters.n,
    )
    canopy_state = canopygrid.compute_initial_state(params.canopygrid)
    canopy_state = canopygrid.update_amax(stand_state.nut_stat, canopy_state)
    moss_state = mosslayer.compute_initial_state(params.mosslayer, constants.mosslayer)
    peat_T_state = temperature.compute_initial_state(constants.temperature)
    strip_state = strip.compute_initial_state(params.strip, constants.strip)
    gv_state = gvegetation.compute_initial_state(params.gvegetation, constants.gvegetation)
    esom_mass = esom.compute_initial_state(params.esom.mass, constants.esom.mass)
    esom_N = esom.compute_initial_state(params.esom.n, constants.esom.n)
    esom_P = esom.compute_initial_state(params.esom.p, constants.esom.p)
    esom_K = esom.compute_initial_state(params.esom.k, constants.esom.k)
    return AllState(
        daily=DailyState(
            canopy=canopy_state, moss=moss_state,
            strip=strip_state, peat_T=peat_T_state,
        ),
        annual=AnnualState(
            stand=stand_state, stand_outputs=stand_out, gv=gv_state,
            esom_mass=esom_mass, esom_N=esom_N, esom_P=esom_P, esom_K=esom_K,
        ),
    )
```

### Step 3: Write `_run_daily_step()`

This function only touches `DailyState`. The annual fields it needs (`hdom`, `leafarea`) are passed explicitly as parameters — they are bound via `functools.partial` before the inner `scan`, so they remain constant across all days in a year.

**Test** (`tests/test_susi_main_daily.py`):
- `_run_daily_step` with known inputs produces expected outputs
- All fields of returned `DailyOutputs` are populated (non-NaN, correct shapes)
- Returned `DailyState` has updated fields
- `# TODO: dwt frozen` — verify `rew_drylimit` uses initial `dwt`, not `strip_state.H - ele`

**Implementation** (`src/supersusi/core/susi_main.py`):
```python
def _run_daily_step(
    daily: DailyState,
    forcing: DailyForcing,
    hdom: np.ndarray,
    leafarea: np.ndarray,
    params: ModuleParams,
    constants: ModuleComputedConstants,
    buffer: strip.NumericalBuffer,  # impure: mutated in-place
) -> tuple[DailyState, DailyOutputs]:

    # TODO: dwt frozen at initial value (legacy behavior)
    # dwt = daily.strip.H - constants.strip.ele  # ← should be this
    dwt = constants.strip.initial_h * np.ones(constants.strip.n)  # ← current behavior
    rew = rew_drylimit(dwt)

    # 1. Canopy hydrology
    cpy_inputs = canopygrid.assemble_inputs(
        WeatherForcings(T=forcing.T, Prec=forcing.Prec,
                        Rg=forcing.Rg, Par=forcing.Par, VPD=forcing.VPD),
        hc=hdom,
        LAIconif=leafarea,
        Rew=rew, beta=daily.moss.Ree,
    )
    cpy_state, cpy_out = canopygrid.run_timestep(
        params.canopygrid, cpy_inputs, daily.canopy,
    )

    # 2. Moss interception
    moss_state, moss_interc_out = mosslayer.run_interception(
        constants.mosslayer,
        mosslayer.assemble_interception_inputs(
            potinf=cpy_out.potinf, evap=cpy_out.efloor,
        ),
        daily.moss,
    )

    # 3. Coupling: water available for soil
    delta = moss_interc_out.potinf - cpy_out.transpi

    # 4. Strip exfil — cap by air volume
    exfil_out = strip.compute_exfil(
        daily.strip, constants.strip,
        forcing.h0ts_west, forcing.h0ts_east, delta,
    )

    # 5. Moss return flow
    moss_state, moss_rf_out = mosslayer.run_returnflow(
        params.mosslayer, constants.mosslayer,
        mosslayer.assemble_returnflow_inputs(
            rflow=exfil_out.exfil,
            interception_mbe=moss_interc_out.mbe,
        ),
        moss_state,
    )

    # 6. Strip PDE
    strip_state, ts_out = strip.run_timestep(
        params.strip, constants.strip, daily.strip,
        strip.assemble_timestep_inputs(
            h0ts_west=forcing.h0ts_west, h0ts_east=forcing.h0ts_east,
            exfil_out=exfil_out, moss_rf_out=moss_rf_out,
        ),
        buffer=buffer,
    )

    # 7. Peat temperature
    peat_T_state, temp_out = temperature.run_timestep(
        params.temperature, constants.temperature,
        temperature.assemble_inputs(
            T_air=forcing.T, swe=cpy_out.swe, efloor=cpy_out.efloor,
        ),
        daily.peat_T,
    )

    new_daily = replace(daily,
        canopy=cpy_state, moss=moss_state,
        strip=strip_state, peat_T=peat_T_state,
    )
    return new_daily, DailyOutputs(
        wtd=strip_state.H - constants.strip.ele,
        afp=ts_out.afp,
        T_soil_hydro=temp_out.T_soil_hydro,
        delta=delta,
        total_runoff=ts_out.roff + np.mean(moss_rf_out.surface_runoff),
        surface_runoff=moss_rf_out.surface_runoff,
        interc=cpy_out.interc,
        evap=cpy_out.evap,
        et=cpy_out.et,
        transpi=cpy_out.transpi,
        efloor=cpy_out.efloor,
        swe=cpy_out.swe,
    )
```

### Step 4: Write `_run_annual_step()`

This function orchestrates one year: runs the inner daily scan via `partial`, then executes the annual biogeochemistry.

The inner scan uses `functools.partial` to bind the values that are constant across all days in the year. The resulting callable has exactly the `(carry, xs) -> (carry, ys)` signature `lax.scan` expects — no lambda or closure wrapper needed.

**Test** (`tests/test_susi_main_year.py`):
- `_run_annual_step` returns updated `AllState` with modified annual fields
- `AnnualOutputs.daily` has shape `(MAX_DAYS, ...)` with zeros in padding region
- `canopygrid.update_amax` runs before daily loop (verify `canopy.amax` changes)
- Cutting branch: when `do_cutting=True`, `stand_out` reflects cutting
- Fertilization, ESOM, methane outputs are populated

**Implementation** (`src/supersusi/core/susi_main.py`):
```python
from functools import partial

MAX_DAYS = 366

def _run_annual_step(
    state: AllState,
    year: AnnualForcing,
    params: ModuleParams,
    constants: ModuleComputedConstants,
    buffer: strip.NumericalBuffer,
) -> tuple[AllState, AnnualOutputs]:

    # ── Pre-daily: update canopy amax from nutrient status ────────
    canopy_state = canopygrid.update_amax(state.annual.stand.nut_stat, state.daily.canopy)

    # ── Inner scan over padded days ──────────────────────────────
    daily_step = partial(
        _run_daily_step,
        hdom=state.annual.stand_outputs.hdom,
        leafarea=state.annual.stand_outputs.leafarea,
        params=params,
        constants=constants,
        buffer=buffer,
    )

    day_forcings = jnp.stack([
        year.daily_T, year.daily_Prec, year.daily_Rg,
        year.daily_Par, year.daily_VPD, year.h0ts_west, year.h0ts_east,
    ], axis=-1)  # (MAX_DAYS, 7)

    def padded_step(daily_state, xs):
        forcing, valid = xs
        def _real(_daily):
            # Replace canopy with updated amax on first valid day
            s = replace(_daily, canopy=canopy_state)
            return daily_step(s, forcing)
        def _noop(_daily):
            return _daily, DailyOutputs(...)
        return lax.cond(valid, _real, _noop, daily_state)

    valid_mask = jnp.arange(MAX_DAYS) < year.valid_days
    daily, stacked_daily = jax.lax.scan(
        padded_step, state.daily, (day_forcings, valid_mask),
    )

    # ── Aggregate daily outputs ──────────────────────────────────
    ndays = year.valid_days
    wtd_yr = stacked_daily.wtd[:ndays]         # (ndays, n)
    afp_yr = stacked_daily.afp[:ndays]         # (ndays, n)
    temp_yr = stacked_daily.T_soil_hydro[:ndays]  # (ndays, n_hydro)
    swe_yr = stacked_daily.swe[:ndays]         # (ndays, n)

    mean_dwt = wtd_yr.mean(axis=0)             # (n,)
    strip_diag = strip.compute_residence_time(params.strip, constants.strip, mean_dwt)

    # ── Annual biogeochemistry ──────────────────────────────────

    # Heterotrophic respiration
    n_days_rhet, Rhet_mean, Rhet_sum = heterotrophic_respiration_yr(
        temp_yr[:, 0], wtd_yr, state.annual.stand_outputs.volume,
        year.calendar_year, constants.stand.dominant.tree_species,
        params.strip.bd_top,
    )

    # Soil CO2 balance
    soil_co2_balance = ojanen_2019(
        constants.stand.dominant.tree_species, wtd_yr, year.calendar_year,
    )

    # Ground vegetation
    gv_state, gv_out = gvegetation.run_timestep(
        params.gvegetation, constants.gvegetation,
        gvegetation.assemble_inputs(
            ts=year.temp_sum,
            vol=state.annual.stand_outputs.volume,
            stems=state.annual.stand_outputs.stems,
            ba=state.annual.stand_outputs.basalarea,
            age=...,  # from susi_params
        ),
        state.annual.gv,
    )

    # Stand growth
    weather_df = pd.DataFrame({...}, index=pd.date_range(...))  # built from year.daily_*
    stand_inputs = stand.Inputs(
        photopara=..., forc=weather_df, wt=wtd_yr, afp=afp_yr,
        n_supply=np.zeros(n), p_supply=np.zeros(n), k_supply=np.zeros(n),
        groundvegetation_outputs=gv_out,
        previous_nut_stat=state.annual.stand.previous_nut_stat,
        calendar_year=year.calendar_year,
    )
    stand_state, stand_out, dom_out, sub_out, under_out = stand.grow_stand(
        state.annual.stand, constants.stand, stand_inputs,
    )

    # Cutting (conditional)
    stand_state, stand_out, cut_out = (
        stand.cut_stand(stand_state, constants.stand, stand_out, stand_inputs)
        if year.do_cutting
        else (stand_state, stand_out, _empty_cut_outputs(n))
    )

    # Fertilization
    _, fert_out = fertilization.run_timestep(
        params.fertilization, constants.fertilization,
        fertilization.assemble_inputs(params.fertilization, year.calendar_year),
    )

    # ESOM pH update
    pH_inc = fert_out.pH_increment
    state_mass = replace(state.annual.esom_mass, pH=esom.update_soil_pH(state.annual.esom_mass.pH, params.esom.mass, pH_inc))
    state_N = replace(state.annual.esom_N, pH=esom.update_soil_pH(state.annual.esom_N.pH, params.esom.n, pH_inc))
    state_P = replace(state.annual.esom_P, pH=esom.update_soil_pH(state.annual.esom_P.pH, params.esom.p, pH_inc))
    state_K = replace(state.annual.esom_K, pH=esom.update_soil_pH(state.annual.esom_K.pH, params.esom.k, pH_inc))

    # ESOM run_yr x4 (mass, N, P, K)
    inputs_mass = esom.assemble_inputs(
        tair_ts=year.daily_T[:ndays],
        tp_top_ts=temp_yr[:, 2], tp_middle_ts=temp_yr[:, 8], tp_bottom_ts=temp_yr[:, 9],
        water_tables=wtd_yr, nonwoodylitter=..., woodylitter=...,
    )
    state_mass, yr_mass = esom.run_yr(params.esom.mass, constants.esom.mass, inputs_mass, state_mass)
    # ... same for N, P, K ...

    doc_export = esom.compose_export(yr_mass.daily_cumulative_out, strip_diag, temp_yr[:, 2], n)

    # Methane
    _, ch4_out = methane.run_timestep(
        inputs=methane.assemble_inputs(year=year.calendar_year, wtd=wtd_yr),
    )

    # Update nutrient status
    nut_inputs = replace(stand_inputs,
        n_supply=yr_N.out_root_lyr + depoN + fert_out.nutrient_release["N"],
        p_supply=yr_P.out_root_lyr + depoP + fert_out.nutrient_release["P"],
        k_supply=yr_K.out_root_lyr + depoK + fert_out.nutrient_release["K"],
    )
    stand_state = stand.update_nutrient_status(stand_state, stand_out, nut_inputs)

    new_annual = replace(state.annual,
        stand=stand_state, stand_outputs=stand_out,
        gv=gv_state, esom_mass=state_mass, esom_N=state_N,
        esom_P=state_P, esom_K=state_K,
    )
    new_state = replace(state, daily=daily, annual=new_annual)
    return new_state, AnnualOutputs(
        daily=stacked_daily,
        stand=stand_out, stand_dom=dom_out, stand_sub=sub_out, stand_under=under_out,
        gv=gv_out,
        esom_mass=yr_mass, esom_N=yr_N, esom_P=yr_P, esom_K=yr_K,
        methane=ch4_out, fertilization=fert_out,
        Rhet=Rhet_sum, soil_co2_balance=soil_co2_balance, doc_export=doc_export,
        strip_diag=strip_diag,
    )
```

### Step 5: Write `_build_annual_forcings()`

**Test** (`tests/test_susi_main_forcings.py`):
- `_build_annual_forcings` returns correct number of years
- Each `AnnualForcing.valid_days` matches actual days in year
- Ditch depths match `drain_depth_development` output
- Padded region (days > valid_days) is zero

**Implementation** (`src/supersusi/core/susi_main.py`):
```python
def _build_annual_forcings(
    susi_params: SusiParams,
    weather_data: pd.DataFrame,
) -> list[AnnualForcing]:
    ...
```
Note: before JAX conversion, `_build_annual_forcings` returns a `list[AnnualForcing]` which is iterated over in a Python `for` loop. At JAX conversion, this becomes a pytree of stacked arrays to pass to `lax.scan`.

### Step 6: Write `_write_outputs()` and rewrite `run()`

**Test** (`tests/test_susi_main_run.py`):
- `run()` completes without error with test data
- `SimulationOutput` contains all expected fields
- Output netcdf file is created (if IO is tested)
- Compare output values against golden reference (if available)

**Implementation** (`src/supersusi/core/susi_main.py`):
```python
def _write_outputs(
    simulation_params: SimulationParams,
    constants: ModuleComputedConstants,
    all_annual: AnnualOutputs,
) -> None:
    """Imperative shell — IO, logging, visualization."""
    out = Outputs(...)
    out.initialize(strip_constants=constants.strip)
    for i, year_out in enumerate(all_annual.annual):
        out.write_stand(...)
        out.write_canopy_layer(...)
        # ... all existing write calls ...
    out.close()


def run(simulation_params: SimulationParams) -> SimulationOutput:
    susi_params = simulation_params.susi_params
    simulation_metadata = simulation_params.metadata

    _create_output_folder(simulation_metadata)

    weather_data = read_FMI_weather(
        ID=0,
        start_date=susi_params.simulation_config.start_date,
        end_date=susi_params.simulation_config.end_date,
        sourcefile=susi_params.weather_parameters.FMI_weather_filepath,
    )

    params = _build_params(susi_params)
    constants = _compute_constants(params, susi_params, weather_data)
    annual_forcings = _build_annual_forcings(susi_params, weather_data)
    init_state = _init_state(params, constants, susi_params)

    buffer = strip.make_numerical_buffer(params.strip)

    # Pre-JAX: Python for loop over annual_forcings.
    # JAX target: jax.lax.scan(_run_annual_step, init_state, annual_forcings,
    #                           args=(params, constants, buffer))
    state = init_state
    all_annual = []
    for yr in annual_forcings:
        state, annual_out = _run_annual_step(state, yr, params, constants, buffer)
        all_annual.append(annual_out)

    _write_outputs(simulation_params, constants, all_annual)

    return SimulationOutput(annual=all_annual, final_state=state)
```

### Step 7: Golden file regression test

After all implementation steps, verify bitwise-equivalent outputs against `tests/golden_file_test/golden_susi.nc`:

**Test** (`tests/golden_file_test/test_supersusi_golden_file.py`):
- Runs the full simulation with `golden_supersusi_test.PARAMETERS`
- Compares every variable in the output netCDF against the golden reference via `np.allclose` (rtol=1e-5, atol=1e-5, `equal_nan=True`)
- Fails on any variable mismatch

This is the definitive regression check. All unit tests (Steps 1-6) are intermediate safety nets; Step 7 validates end-to-end correctness.

---

## Module changes summary

| Module | Change | Status | Files |
|---|---|---|---|
| `susi_main` | Full rewrite (dataclasses + orchestrator) | **Active** | `susi_main.py`, `tests/test_susi_main_*.py` |
| `susi_utils` | `heterotrophic_respiration_yr`, `ojanen_2019`: DataFrame → array | **Deferred** | `susi_utils.py`, `tests/test_susi_utils_array.py` |
| `stand` | `Inputs`: DataFrame → array fields | **Deferred** | `stand.py`, `tests/test_stand_inputs.py` |
| `methane` | `assemble_inputs`: DataFrame → array | **Deferred** | `methane.py`, `tests/test_methane.py` |

## What stays unchanged

- `_build_params()` — already functional
- `_compute_constants()` — already functional
- Module-level code (`canopygrid.py`, `mosslayer.py`, `strip.py`, `temperature.py`, `esom.py`, `fertilization.py`, `gvegetation.py`) — already functional
- `strip.NumericalBuffer` — deferred, keeps in-place mutation
- IO code — stays imperative

## Known issues (deferred)

1. **dwt bug**: `rew_drylimit` uses frozen initial water table. Preserved as legacy behavior.
2. **NumericalBuffer impurity**: `strip.run_timestep` mutates buffer in-place. `_run_daily_step` is impure because of this.
3. **scipy in modules**: `temperature.py` uses `scipy.linalg.solve`, `esom.py` uses `scipy.sparse`. Not JAX-compatible. Deferred to future JAX conversion step.
4. **`stand.Inputs` Any types**: `photopara` and `groundvegetation_outputs` remain `Any` for now. Should become concrete dataclasses before JAX conversion.
5. Turn dataframes into numpy arrays.

---

## JAX gradient example (future)

```python
def simulate_and_loss(params, forcings):
    state = _init_simulation_state(params, ...)
    final, all_out = jax.lax.scan(_run_annual_step, state, forcings)
    return jnp.sum(all_out.methane.ch4)

grad_fn = jax.grad(simulate_and_loss)
grads = grad_fn(params, forcings)
```

# SuperSUSI scan architecture

A design for a functional, JAX-ready `susi_main.py` orchestrator.

---

## Disclaimers

> **JAX status:** JAX conversion is deferred to a future phase. The architecture is
> designed for a smooth JAX transition (two nested `jax.lax.scan` over year, day),
> but all loops in this phase are plain Python `for` loops. This means:
> no JAX primitives, `strip.NumericalBuffer` impurity is harmless, scipy usage in
> modules is fine.

> **Scenario loop:** Eliminated. Single scenario execution. Output IO uses a
> hard-coded `n_ditch_scen=0` and keeps a leading `(1, ...)` dimension in NetCDF
> files for backward compatibility with existing IO.

---

## Background
The simulation has two timestep loops: daily and yearly.
The daily simulations are performed at the start of every year in the yearly loop.

## Core principles
Dataclasses are pure structs: they must have zero methods.

Architecture: two loops over (year, day), eventually becoming two nested `jax.lax.scan`.
All accumulation is scan's return value.

```
run(params, forcings)
  ├── _build_params()           # already functional
  ├── _compute_constants()      # already functional
  ├── _init_state()             # builds AllState from params + constants
  ├── for yr in annual_forcings:
  │     ├── inner loop: _run_daily_step (hydrology + temperature)
  │     └── annual biogeochemistry (stand, esom, fertilization, methane)
  └── _write_outputs()          # imperative shell, outside loops
```
---

## Decisions

- **NumericalBuffer**: Deferred. `strip.NumericalBuffer` keeps in-place mutation.
  `_run_daily_step` is impure because of this. A future step will replace the
  strip solver.
- **dwt bug**: Not fixed. The current code never feeds `strip_state.H - constants.ele`
  back into `rew_drylimit`. The functional rewrite preserves this legacy behavior.
  Mark it with `# BUG: dwt frozen at initial value`.
- **Scenario loop**: Eliminated. Single scenario execution. Output arrays keep a
  leading `(1, ...)` dimension for backward compatibility with existing IO.
- **pandas DataFrames**: Deferred. JAX cannot handle them. They must be eliminated
  from all traced code paths before JAX conversion. IO code (imperative shell) may
  still use pandas. The `_run_annual_step` function keeps DataFrame construction
  for `stand.Inputs`, `methane.assemble_inputs`, `heterotrophic_respiration_yr`,
  and `ojanen_2019` — these will be replaced in a separate phase.
- **No padding**: Since the daily loop is a Python `for`, each year's arrays have
  length `valid_days` (365 or 366). No `MAX_DAYS` constant is needed.

---

## Orchestrator dataclasses

State and Output is separated with the same logic of SuperSUSI_functional_architecture.md:
State is the `carry` in `jax.lax.scan`, Outputs are the variables not used in the next scan loop.

Because we have 2 loops (daily and annual), we need:

- `DailyState` (inner loop carry)
- `AnnualState` (outer loop carry, read-only during inner loop)
- `AllState` (top-level carry, bundles both)
- `DailyForcing`
- `DailyOutputs`
- `AnnualForcing`
- `AnnualOutputs`
- `SimulationOutput`


```python
@dataclass(frozen=True)
class DailyState:
    """Evolving inside the inner loop — updated every day."""
    canopy: canopygrid.State
    moss: mosslayer.State
    strip: strip.State
    peat_T: temperature.State


@dataclass(frozen=True)
class AnnualState:
    """Evolving inside the outer loop — updated once per year.
    Read-only during the inner daily loop.
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
    """Top-level carry for the outer loop."""
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
    """Per-day outputs — inner loop's ys pytree."""
    wtd: np.ndarray                  # (n,) — strip_state.H - constants.ele
    afp: np.ndarray                  # (n,) — from strip
    T_soil_hydro: np.ndarray         # (n_hydro,) — from temperature, first n_layers_hydro
    delta: np.ndarray                # (n,)  potinf - transpi, coupling
    total_runoff: float              # ts_out.roff + mean(surface_runoff)
    surface_runoff: np.ndarray       # (n,) — from moss_returnflow
    interc: np.ndarray               # (n,) — from canopy
    evap: np.ndarray                 # (n,) — from canopy
    et: np.ndarray                   # (n,) — from canopy
    transpi: np.ndarray              # (n,) — from canopy
    efloor: np.ndarray               # (n,) — from canopy (pre-moss, matches current output)
    swe: np.ndarray                  # (n,) — from canopy


@dataclass(frozen=True)
class AnnualForcing:
    """All inputs for one year — outer loop iterates over these.

    Array shapes are per-year (variable length, valid_days = 365 or 366).
    No padding — the daily loop iterates exactly valid_days times.
    """
    # Daily arrays (used by annual biogeochemistry)
    daily_T: np.ndarray              # (days,) air temperature
    daily_Rg: np.ndarray             # (days,) global radiation
    daily_VPD: np.ndarray            # (days,) VPD

    # Boundary conditions for the inner daily loop
    daily_Prec: np.ndarray           # (days,) precipitation
    daily_Par: np.ndarray            # (days,) PAR
    h0ts_west: np.ndarray            # (days,) west ditch depth per day
    h0ts_east: np.ndarray            # (days,) east ditch depth per day

    valid_days: int                  # 365 or 366
    calendar_year: int               # for phenology, management, stand inputs
    temp_sum: np.ndarray             # (n,) temperature sum for ground vegetation

    # Annual management flags
    do_cutting: bool
    cutting_to_ba: float


@dataclass(frozen=True)
class AnnualOutputs:
    """Per-year outputs — includes stacked daily outputs for IO + biogeochemistry."""
    daily: DailyOutputs              # stacked → (days, *)
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


@dataclass(frozen=True)
class SimulationOutput:
    """Top-level return from run()."""
    annual: list[AnnualOutputs]
    final_state: AllState


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

These will be tackled in a separate phase before JAX conversion. The `_run_annual_step` code below uses DataFrames as-is.

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

Note: the initial canopy layer outputs (`dom_out`, `sub_out`, `under_out`) and the
year-0 gv outputs are not stored in `AllState` — they are only needed by `_write_outputs`
for the year 0 IO, and are recomputed there by calling `stand.compute_initial_state`
and `gvegetation.run_timestep` with the initial state.

### Step 3: Write `_run_daily_step()`

This function only touches `DailyState`. The annual fields it needs (`hdom`, `leafarea`) are passed explicitly as parameters — they are bound before the inner loop, so they remain constant across all days in a year.

`DailyOutputs.efloor` stores the pre-moss efloor (from canopy output), matching the current code where `efloors[...]` is filled before moss interception modifies the variable. The temperature module receives the **post-moss** efloor (`moss_interc_out.evap`) to match the current code's behavior at lines 549-551, 614-616.

**Test** (`tests/test_susi_main_daily.py`):
- `_run_daily_step` with known inputs produces expected outputs
- All fields of returned `DailyOutputs` are populated (non-NaN, correct shapes)
- Returned `DailyState` has updated fields
- Temperature module receives moss-modified efloor, not raw canopy efloor
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
        hc=hdom, LAIconif=leafarea, Rew=rew, beta=daily.moss.Ree,
    )
    cpy_state, cpy_out = canopygrid.run_timestep(
        params.canopygrid, cpy_inputs, daily.canopy,
    )

    # 2. Moss interception (modifies potinf, evap)
    moss_state, moss_interc_out = mosslayer.run_interception(
        constants.mosslayer,
        mosslayer.assemble_interception_inputs(
            potinf=cpy_out.potinf, evap=cpy_out.efloor,
        ),
        daily.moss,
    )

    # 3. Coupling: water available for soil (uses moss-modified potinf)
    moss_efloor = moss_interc_out.evap
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
            rflow=exfil_out.exfil, interception_mbe=moss_interc_out.mbe,
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

    # 7. Peat temperature (uses moss-modified efloor)
    peat_T_state, temp_out = temperature.run_timestep(
        params.temperature, constants.temperature,
        temperature.assemble_inputs(
            T_air=forcing.T, swe=cpy_out.swe, efloor=moss_efloor,
        ),
        daily.peat_T,
    )

    new_daily = replace(daily,
        canopy=cpy_state, moss=moss_state,
        strip=strip_state, peat_T=peat_T_state,
    )
    # DailyOutputs.efloor = pre-moss canopy efloor (matches current output arrays)
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

This function orchestrates one year: runs the inner daily loop, then executes the annual biogeochemistry.

The inner loop is a plain Python `for` over `year.valid_days` days. Annual biogeochemistry uses DataFrames as-is (deferred; the code mirrors the current `susi_main.py` structure).

`_run_annual_step` also needs access to a few extra pieces from `susi_params` that have not been split into `ModuleParams`/`ModuleComputedConstants` yet. Rather than threading `susi_params` through, these are added to `ModuleComputedConstants` (see Step 1 additions):
- `constants.age: np.ndarray` — tree age per column (`susi_params.site_parameters.age["dominant"]`)
- `constants.depoN / depoP / depoK: float` — atmospheric deposition
- `constants.spara` — the site parameters Pydantic model (for `heterotrophic_respiration_yr`, `ojanen_2019`)

> **Note on spara:** This is a Pydantic model, not a dataclass. It violates the
> "pure struct" principle but is a temporary bridge. It will be replaced once
> `susi_utils.py` functions are converted to accept arrays.

**Test** (`tests/test_susi_main_year.py`):
- `_run_annual_step` returns updated `AllState` with modified annual fields
- `AnnualOutputs.daily` has shape `(year.valid_days, ...)`
- `canopygrid.update_amax` runs before daily loop (verify `canopy.amax` changes)
- Cutting branch: when `do_cutting=True`, `stand_out` reflects cutting
- Fertilization, ESOM, methane outputs are populated

**Implementation** (`src/supersusi/core/susi_main.py`):
```python
def _run_annual_step(
    state: AllState,
    year: AnnualForcing,
    params: ModuleParams,
    constants: ModuleComputedConstants,
    buffer: strip.NumericalBuffer,
) -> tuple[AllState, AnnualOutputs]:

    ndays = year.valid_days
    n = params.strip.n
    n_hydro = params.temperature.n_layers_hydro

    # ── Pre-daily: update canopy amax from nutrient status ────────
    canopy_state = canopygrid.update_amax(state.annual.stand.nut_stat, state.daily.canopy)
    daily = replace(state.daily, canopy=canopy_state)

    # ── Pre-allocate daily output arrays ──────────────────────────
    wtd_yr = np.zeros((ndays, n))
    afp_yr = np.zeros((ndays, n))
    T_soil_hydro_yr = np.zeros((ndays, n_hydro))
    delta_yr = np.zeros((ndays, n))
    total_runoff_yr = np.zeros(ndays)
    surface_runoff_yr = np.zeros((ndays, n))
    interc_yr = np.zeros((ndays, n))
    evap_yr = np.zeros((ndays, n))
    et_yr = np.zeros((ndays, n))
    transpi_yr = np.zeros((ndays, n))
    efloor_yr = np.zeros((ndays, n))
    swe_yr = np.zeros((ndays, n))

    # ── Inner daily loop (Python for) ─────────────────────────────
    for dd in range(ndays):
        forcing = DailyForcing(
            T=year.daily_T[dd], Prec=year.daily_Prec[dd],
            Rg=year.daily_Rg[dd], Par=year.daily_Par[dd],
            VPD=year.daily_VPD[dd],
            h0ts_west=year.h0ts_west[dd], h0ts_east=year.h0ts_east[dd],
        )
        daily, daily_out = _run_daily_step(
            daily, forcing,
            hdom=state.annual.stand_outputs.hdom,
            leafarea=state.annual.stand_outputs.leafarea,
            params=params, constants=constants, buffer=buffer,
        )
        wtd_yr[dd] = daily_out.wtd
        afp_yr[dd] = daily_out.afp
        T_soil_hydro_yr[dd] = daily_out.T_soil_hydro
        delta_yr[dd] = daily_out.delta
        total_runoff_yr[dd] = daily_out.total_runoff
        surface_runoff_yr[dd] = daily_out.surface_runoff
        interc_yr[dd] = daily_out.interc
        evap_yr[dd] = daily_out.evap
        et_yr[dd] = daily_out.et
        transpi_yr[dd] = daily_out.transpi
        efloor_yr[dd] = daily_out.efloor
        swe_yr[dd] = daily_out.swe

    # ── Stack into DailyOutputs ───────────────────────────────────
    stacked_daily = DailyOutputs(
        wtd=wtd_yr, afp=afp_yr, T_soil_hydro=T_soil_hydro_yr,
        delta=delta_yr, total_runoff=total_runoff_yr,
        surface_runoff=surface_runoff_yr, interc=interc_yr,
        evap=evap_yr, et=et_yr, transpi=transpi_yr,
        efloor=efloor_yr, swe=swe_yr,
    )

    # ── Aggregate daily outputs ──────────────────────────────────
    strip_diag = strip.compute_residence_time(
        params.strip, constants.strip, wtd_yr.mean(axis=0),
    )

    # ── Annual biogeochemistry ──────────────────────────────────
    # (DataFrames preserved — deferred step)

    sday = datetime.datetime(year.calendar_year, 1, 1)
    dfwt = pd.DataFrame(wtd_yr, index=pd.date_range(sday, periods=ndays))
    dfafp = pd.DataFrame(afp_yr, index=pd.date_range(sday, periods=ndays))
    df_peat_temps = pd.DataFrame(
        T_soil_hydro_yr,
        index=pd.date_range(sday, periods=ndays),
    )

    # Heterotrophic respiration
    _, _co2, Rhet = heterotrophic_respiration_yr(
        df_peat_temps, year.calendar_year, dfwt,
        state.annual.stand_outputs.volume, constants.spara,
    )

    # Soil CO2 balance
    soil_co2_balance = ojanen_2019(
        constants.spara, year.calendar_year, dfwt,
    )

    # Ground vegetation
    gv_state, gv_out = gvegetation.run_timestep(
        params.gvegetation, constants.gvegetation,
        gvegetation.assemble_inputs(
            ts=year.temp_sum,
            vol=state.annual.stand_outputs.volume,
            stems=state.annual.stand_outputs.stems,
            ba=state.annual.stand_outputs.basalarea,
            age=constants.age,
        ),
        state.annual.gv,
    )

    # Stand growth
    weather_yr = weather_data.loc[str(year.calendar_year)]
    stand_inputs = stand.Inputs(
        photopara=susi_params.photo_parameters,   # TODO: wire through params
        forc=weather_yr, wt=dfwt, afp=dfafp,
        n_supply=np.zeros(n), p_supply=np.zeros(n), k_supply=np.zeros(n),
        groundvegetation_outputs=gv_out,
        previous_nut_stat=state.annual.stand.previous_nut_stat,
        calendar_year=year.calendar_year,
    )
    stand_state, stand_out, dom_out, sub_out, under_out = stand.grow_stand(
        state.annual.stand, constants.stand, stand_inputs,
    )

    # Cutting (conditional)
    if year.do_cutting:
        cut_inputs = replace(stand_inputs, cutting_to_ba=year.cutting_to_ba)
        stand_state, stand_out, _ = stand.cut_stand(
            stand_state, constants.stand, stand_out, cut_inputs,
        )

    # Fertilization
    _, fert_out = fertilization.run_timestep(
        params.fertilization, constants.fertilization,
        fertilization.assemble_inputs(params.fertilization, year.calendar_year),
    )

    # ESOM pH update + run_yr x4 (mass, N, P, K)
    # (Identical structure to current susi_main.py — see lines 771-923)
    #   pH_inc = fert_out.pH_increment
    #   state_mass = replace(state.annual.esom_mass,
    #       pH=esom.update_soil_pH(state.annual.esom_mass.pH, params.esom.mass, pH_inc))
    #   ... same for N, P, K ...
    #   inputs_mass = esom.assemble_inputs(...)
    #   state_mass, yr_out_mass = esom.run_yr(...)
    #   ... same for N, P, K ...
    #   doc_export = esom.compose_export(...)

    # Nutrient status update
    nut_inputs = replace(stand_inputs,
        n_supply=yr_out_N.out_root_lyr + constants.depoN + fert_out.nutrient_release["N"],
        p_supply=yr_out_P.out_root_lyr + constants.depoP + fert_out.nutrient_release["P"],
        k_supply=yr_out_K.out_root_lyr + constants.depoK + fert_out.nutrient_release["K"],
    )
    stand_state = stand.update_nutrient_status(stand_state, stand_out, nut_inputs)

    # Methane
    _, ch4_out = methane.run_timestep(
        inputs=methane.assemble_inputs(year=year.calendar_year, dfwt=dfwt),
    )

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
        esom_mass=yr_out_mass, esom_N=yr_out_N, esom_P=yr_out_P, esom_K=yr_out_K,
        methane=ch4_out, fertilization=fert_out,
        Rhet=Rhet, soil_co2_balance=soil_co2_balance, doc_export=doc_export,
        strip_diag=strip_diag,
    )
```

### Step 5: Write `_build_annual_forcings()`

**Test** (`tests/test_susi_main_forcings.py`):
- Returns correct number of years
- Each `AnnualForcing.valid_days` matches actual days in year (365 or 366)
- Ditch depths match `drain_depth_development` output
- Year slicing is contiguous (last day of year N + 1 = first day of year N+1)
- `do_cutting` is `True` only on the cutting year

**Implementation** (`src/supersusi/core/susi_main.py`):
```python
def _build_annual_forcings(
    susi_params: SusiParams,
    weather_data: pd.DataFrame,
    temp_sum: np.ndarray,
) -> list[AnnualForcing]:
    """
    Build per-year forcing bundles from the full weather dataframe.

    Single scenario: picks the first ditch depth parameters.
    Chunks weather data and ditch depths into per-year numpy arrays.
    """
    n_days = (
        susi_params.simulation_config.end_date
        - susi_params.simulation_config.start_date
    ).days + 1

    # Single scenario: first ditch depth parameters
    h0ts_west = drain_depth_development(
        n_days,
        susi_params.site_parameters.ditch_depth_west[0],
        susi_params.site_parameters.ditch_depth_20y_west[0],
    )
    h0ts_east = drain_depth_development(
        n_days,
        susi_params.site_parameters.ditch_depth_east[0],
        susi_params.site_parameters.ditch_depth_20y_east[0],
    )

    forcings: list[AnnualForcing] = []
    d = 0
    for calendar_year in range(
        susi_params.simulation_config.start_date.year,
        susi_params.simulation_config.end_date.year + 1,
    ):
        days = (
            datetime.datetime(calendar_year, 12, 31)
            - datetime.datetime(calendar_year, 1, 1)
        ).days + 1
        w = weather_data.iloc[d : d + days]

        forcings.append(AnnualForcing(
            daily_T=w["T"].values,
            daily_Rg=w["Rg"].values,
            daily_VPD=w["VPD"].values,
            daily_Prec=w["Prec"].values,
            daily_Par=w["Par"].values,
            h0ts_west=h0ts_west[d : d + days],
            h0ts_east=h0ts_east[d : d + days],
            valid_days=days,
            calendar_year=calendar_year,
            temp_sum=temp_sum,
            do_cutting=(calendar_year == susi_params.site_parameters.cutting_yr),
            cutting_to_ba=susi_params.site_parameters.cutting_to_ba,
        ))
        d += days

    return forcings
```

### Step 6: Write `_write_outputs()` and rewrite `run()`

`_write_outputs` handles all IO in the imperative shell. It hard-codes `n_ditch_scen=0`
for the single scenario. Year 0 initial state is written from `stand.compute_initial_state`
and `init_state`; years 1..N are written from `all_annual`.

`_write_outputs` recomputes the year-0 canopy layer outputs and gv outputs (needed only for
IO) by calling `stand.compute_initial_state` and `gvegetation.run_timestep` with the initial
state. This is acceptable because `_write_outputs` is imperative shell.

**Test** (`tests/test_susi_main_run.py`):
- `run()` completes without error with test data
- `SimulationOutput` contains all expected fields
- Output NetCDF file is created (if IO is tested)
- Compare output values against golden reference (if available)

**Implementation** (`src/supersusi/core/susi_main.py`):
```python
def _write_outputs(
    simulation_params: SimulationParams,
    params: ModuleParams,
    constants: ModuleComputedConstants,
    init_state: AllState,
    all_annual: list[AnnualOutputs],
    weather_data: pd.DataFrame,
) -> None:
    """Imperative shell — IO, logging, visualization.

    Single scenario: n_ditch_scen = 0 everywhere.
    Year 0 state is from init_state (recomputed as needed).
    """
    susi_params = simulation_params.susi_params
    n = susi_params.site_parameters.n
    n_years = len(all_annual)
    n_total_days = (susi_params.simulation_config.end_date
                    - susi_params.simulation_config.start_date).days + 1

    out = Outputs(
        n_scenarios=1,
        n_cols=n,
        n_days=n_total_days,
        n_years=n_years,
        n_layers=susi_params.site_parameters.nLyrs,
        fname=simulation_params.metadata.netcdf_output_filepath,
    )
    out.initialize(strip_constants=constants.strip)

    n_ditch_scen = 0

    # ── Static metadata ────────────────────────────────────────────
    out.write_paras(
        sfc=susi_params.site_parameters.sfc,
        dominant_sp=constants.stand.dominant.tree_species,
        subdominant_sp=constants.stand.subdominant.tree_species,
        under_sp=constants.stand.under.tree_species,
    )
    out.write_scen(
        n_ditch_scen,
        susi_params.site_parameters.ditch_depth_west[0],
        susi_params.site_parameters.ditch_depth_east[0],
    )
    susi_io.print_site_description(susi_params.site_parameters)

    # ── Year 0: initial state ─────────────────────────────────────
    # Recompute initial stand outputs (needed for canopy layer IO)
    stand_state0, stand_out0, dom_out0, sub_out0, under_out0 = (
        stand.compute_initial_state(
            params.stand, constants.stand,
            susi_params.site_parameters.age, n,
        )
    )
    # Year 0 gv (using initial stand outputs)
    temp_sum = _build_annual_forcings(...)  # or passed separately
    gv_out0 = gvegetation.run_timestep(
        params.gvegetation, constants.gvegetation,
        gvegetation.assemble_inputs(
            ts=temp_sum,
            vol=stand_out0.volume,
            stems=stand_out0.stems,
            ba=stand_out0.basalarea,
            age=constants.age,
        ),
        init_state.annual.gv,
    )[1]

    out.write_stand(n_ditch_scen, 0, stand_out0, stand_state0,
                    previous_nut_stat=stand_state0.previous_nut_stat)
    out.write_canopy_layer(n_ditch_scen, 0, "dominant", stand_state0.dominant, dom_out0)
    out.write_canopy_layer(n_ditch_scen, 0, "subdominant", stand_state0.subdominant, sub_out0)
    out.write_canopy_layer(n_ditch_scen, 0, "under", stand_state0.under, under_out0)
    out.write_groundvegetation(n_ditch_scen, 0, init_state.annual.gv, gv_out0)
    out.write_esom(n_ditch_scen, 0, "Mass", init_state.annual.esom_mass, inivals=True)
    out.write_esom(n_ditch_scen, 0, "N", init_state.annual.esom_N, inivals=True)
    out.write_esom(n_ditch_scen, 0, "P", init_state.annual.esom_P, inivals=True)
    out.write_esom(n_ditch_scen, 0, "K", init_state.annual.esom_K, inivals=True)

    # ── Year 1..N: from annual outputs ────────────────────────────
    start = 0
    for i, yr in enumerate(all_annual, start=1):
        ndays = yr.daily.wtd.shape[0]

        out.write_cpy(n_ditch_scen, start, ndays, i,
                      yr.daily.interc, yr.daily.evap, yr.daily.et,
                      yr.daily.transpi, yr.daily.efloor, yr.daily.swe)
        out.write_temperature(n_ditch_scen, start, ndays, yr.daily.T_soil_hydro)

        # Build dataframes for write_strip (matches current API)
        sday = datetime.datetime(..., 1, 1)
        dfwt = pd.DataFrame(yr.daily.wtd, index=pd.date_range(sday, periods=ndays))
        stpout = {
            "dwts": dfwt.values[np.newaxis, ...],          # (1, ndays, n)
            "afps": yr.daily.afp[np.newaxis, ...],
            "deltas": yr.daily.delta[np.newaxis, ...],
            "hts": (yr.daily.wtd + ...)[np.newaxis, ...],  # H = wtd + ele
            "runoff": yr.daily.total_runoff[np.newaxis],
            "runoffwest": ...,                              # not in DailyOutputs
            "runoffeast": ...,
            "surfacerunoff": yr.daily.surface_runoff[np.newaxis, ...],
        }
        out.write_strip(n_ditch_scen, start, ndays, ..., i, dfwt, stpout,
                        susi_params.output_parameters, yr.strip_diag)

        out.write_ojanen(n_ditch_scen, i, yr.Rhet, yr.soil_co2_balance)
        out.write_fertilization(n_ditch_scen, i, yr.fertilization)
        out.write_esom(n_ditch_scen, i, "Mass", ..., yr.esom_mass)
        out.write_esom(n_ditch_scen, i, "N", ..., yr.esom_N)
        out.write_esom(n_ditch_scen, i, "P", ..., yr.esom_P)
        out.write_esom(n_ditch_scen, i, "K", ..., yr.esom_K)
        out.write_methane(n_ditch_scen, i, yr.methane)
        out.write_export(n_ditch_scen, i, yr.doc_export)

        out.write_stand(n_ditch_scen, i, yr.stand, ...,
                        previous_nut_stat=...)
        out.write_canopy_layer(n_ditch_scen, i, "dominant", ..., yr.stand_dom)
        out.write_canopy_layer(n_ditch_scen, i, "subdominant", ..., yr.stand_sub)
        out.write_canopy_layer(n_ditch_scen, i, "under", ..., yr.stand_under)
        out.write_groundvegetation(n_ditch_scen, i, ..., yr.gv)

        out.write_nutrient_balance(n_ditch_scen, i, "N", yr.esom_N,
                                   constants.depoN, ...)
        out.write_nutrient_balance(n_ditch_scen, i, "P", yr.esom_P,
                                   constants.depoP, ...)
        out.write_nutrient_balance(n_ditch_scen, i, "K", yr.esom_K,
                                   constants.depoK, ...)
        out.write_carbon_balance(n_ditch_scen, i, yr.stand, yr.gv,
                                 yr.esom_mass, yr.doc_export, yr.methane)

        start += ndays

    out.close()
```

Note: `run()` needs to pass `temp_sum` to `_write_outputs` (or the forcings). The
simplest approach is to compute `temp_sum` once in `run()` and pass it alongside
the other data. Alternatively, `_write_outputs` can rebuild the forcings — since
it's imperative shell, either is fine.

```python
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
    temp_sum = get_temp_sum(weather_data)
    annual_forcings = _build_annual_forcings(susi_params, weather_data, temp_sum)
    init_state = _init_state(params, constants, susi_params)

    buffer = strip.make_numerical_buffer(params.strip)

    state = init_state
    all_annual: list[AnnualOutputs] = []
    for yr in annual_forcings:
        state, annual_out = _run_annual_step(state, yr, params, constants, buffer)
        all_annual.append(annual_out)

    _write_outputs(simulation_params, params, constants, init_state,
                   all_annual, weather_data)
    write_params_and_metadata(simulation_metadata, susi_params)

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
2. **NumericalBuffer impurity**: `strip.run_timestep` mutates buffer in-place. Harmless with Python for loops; needs fixing before JAX scan.
3. **scipy in modules**: `temperature.py` uses `scipy.linalg.solve`, `esom.py` uses `scipy.sparse`. Not JAX-compatible. Deferred to future JAX conversion step.
4. **`stand.Inputs` Any types**: `photopara` and `groundvegetation_outputs` remain `Any` for now. Should become concrete dataclasses before JAX conversion.
5. **DataFrames in `_run_annual_step`**: `stand.Inputs`, `methane.assemble_inputs`, `heterotrophic_respiration_yr`, `ojanen_2019` still use pandas DataFrames. Deferred to a separate phase.
6. **`spara` in `susi_utils.py`**: `heterotrophic_respiration_yr` and `ojanen_2019` accept a Pydantic model (`spara`). Carried as `constants.spara` for now; will be replaced with individual array parameters when the DataFrame→array conversion happens.

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

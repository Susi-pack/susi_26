# SuperSUSI scan architecture

A design for a differentiable, JAX-compatible `susi_main.py` orchestrator.

---

## Core principle

One `SimulationState`, one `DailyOutputs`, one flat pytree of `YearlyForcing` — nested `jax.lax.scan` over (year, day). Zero methods on dataclasses. All accumulation is `scan`'s return value.

---

## Orchestrator types

```python
@dataclass(frozen=True)
class SimulationState:
    """All evolving state — one ring to rule them all."""
    # Daily-evolving (hydrology + temperature)
    daily_canopy: canopygrid.State
    daily_moss: mosslayer.State
    daily_strip: strip.State
    daily_peat_T: temperature.State

    # Annually-evolving (constant within a year, used by daily loop)
    annual_stand_state: stand.State
    annual_stand_out: stand.Outputs        # hdom, leafarea — read by daily canopy step
    annual_gv: gvegetation.State
    annual_esom_mass: esom.State
    annual_esom_N: esom.State
    annual_esom_P: esom.State
    annual_esom_K: esom.State


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
    """Per-day outputs — scan's ys pytree."""
    strip: strip.TimestepOutputs       # wtd, ht, afp, roffwest, roffeast — enriched
    canopy: canopygrid.Outputs         # interc, evap, et, transpi, efloor, swe
    T_soil_hydro: np.ndarray           # (n_hydro,) — from temperature.Outputs
    delta: np.ndarray                  # (n,)  potinf - transpi, coupling
    total_runoff: float                # ts_out.roff + mean(surface_runoff), coupling
    surface_runoff: np.ndarray         # (n,) — from moss_returnflow, coupling


@dataclass(frozen=True)
class YearlyForcing:
    """All inputs for one year — outer scan iterates over these."""
    daily_forcings: DailyForcing       # each leaf shape (MAX_DAYS, *), padded
    valid_days: int                    # 365 or 366
    calendar_year: int                 # for phenology, management
    weather_yr: np.ndarray             # (MAX_DAYS, 5) — [T, Prec, Rg, Par, VPD]

    # Annual management flags
    do_cutting: bool
    cutting_to_ba: float


@dataclass(frozen=True)
class AnnualOutputs:
    """Per-year outputs — includes stacked daily outputs for IO + biogeochemistry."""
    daily: DailyOutputs                # stacked → (MAX_DAYS, *)
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
    Rhet: np.ndarray                   # (n,)
    soil_co2_balance: np.ndarray       # (n,)
    doc_export: esom.DOCExportOutputs


@dataclass(frozen=True)
class SimulationOutput:
    """Top-level return value."""
    annual: AnnualOutputs              # stacked → (n_years, *)
    final_state: SimulationState
```

---

## Module enrichments

### strip.py

`TimestepOutputs` gains two fields, computed inside `run_timestep()`:

```python
@dataclass(frozen=True)
class TimestepOutputs:
    roff: float
    roffwest: float
    roffeast: float
    air_ratio: np.ndarray          # (n,)
    afp: np.ndarray                # (n,)
    wtd: np.ndarray                # (n,)  H - ele  ← new
    ht: np.ndarray                 # (n,)  H        ← new
```

`run_timestep()` optionally accepts `buffer=None` (allocates internally for JAX path).

### temperature.py

`Outputs` gains one field:

```python
@dataclass(frozen=True)
class Outputs:
    T_soil_hydro: np.ndarray       # (n_hydro,)  T_soil[:n_layers_hydro]  ← new
```

Computed inside `run_timestep()`.

---

## Core step functions

### _run_daily_step — one day of hydrology + temperature

```python
def _run_daily_step(
    state: SimulationState,
    forcing: DailyForcing,
    params: ModuleParams,
    constants: ModuleComputedConstants,
) -> tuple[SimulationState, DailyOutputs]:

    rew = rew_drylimit(state.daily_strip.H - constants.strip.ele)

    # 1. Canopy hydrology
    cpy_inputs = canopygrid.assemble_inputs(
        WeatherForcings(T=forcing.T, Prec=forcing.Prec,
                        Rg=forcing.Rg, Par=forcing.Par, VPD=forcing.VPD),
        hc=state.annual_stand_out.hdom,
        LAIconif=state.annual_stand_out.leafarea,
        Rew=rew, beta=state.daily_moss.Ree,
    )
    cpy_state, cpy_out = canopygrid.run_timestep(
        params.canopygrid, cpy_inputs, state.daily_canopy,
    )

    # 2. Moss interception
    moss_state, moss_interc_out = mosslayer.run_interception(
        constants.mosslayer,
        mosslayer.assemble_interception_inputs(
            potinf=cpy_out.potinf, evap=cpy_out.efloor,
        ),
        state.daily_moss,
    )

    # 3. Coupling: water available for soil
    delta = moss_interc_out.potinf - cpy_out.transpi

    # 4. Strip exfil — cap by air volume
    exfil_out = strip.compute_exfil(
        state.daily_strip, constants.strip,
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
        params.strip, constants.strip, state.daily_strip,
        strip.assemble_timestep_inputs(
            h0ts_west=forcing.h0ts_west, h0ts_east=forcing.h0ts_east,
            exfil_out=exfil_out, moss_rf_out=moss_rf_out,
        ),
    )

    # 7. Peat temperature
    peat_T_state, temp_out = temperature.run_timestep(
        params.temperature, constants.temperature,
        temperature.assemble_inputs(
            T_air=forcing.T, swe=cpy_out.swe, efloor=cpy_out.efloor,
        ),
        state.daily_peat_T,
    )

    new_state = replace(state,
        daily_canopy=cpy_state, daily_moss=moss_state,
        daily_strip=strip_state, daily_peat_T=peat_T_state,
    )
    return new_state, DailyOutputs(
        strip=ts_out, canopy=cpy_out,
        T_soil_hydro=temp_out.T_soil_hydro,
        delta=delta,
        total_runoff=ts_out.roff + np.mean(moss_rf_out.surface_runoff),
        surface_runoff=moss_rf_out.surface_runoff,
    )
```

### _run_year_step — inner scan over days + annual biogeochemistry

```python
def _run_year_step(
    state: SimulationState,
    year: YearlyForcing,
    params: ModuleParams,
    constants: ModuleComputedConstants,
) -> tuple[SimulationState, AnnualOutputs]:

    # ── Inner scan over padded days ──────────────────────────────
    MAX_DAYS = 366

    def _padded_step(state, xs):
        forcing, valid = xs  # valid: bool, False for padding days
        def _real(state):
            return _run_daily_step(state, forcing, params, constants)
        def _noop(state):
            return state, DailyOutputs.zeros_like()
        return lax.cond(valid, _real, _noop, state)

    day_forcings = _pad_to_max(year.daily_forcings, MAX_DAYS)
    valid_mask = np.arange(MAX_DAYS) < year.valid_days

    state, stacked_daily = jax.lax.scan(
        _padded_step, state, (day_forcings, valid_mask),
    )

    # ── Slice off padding ──────────────────────────────────────────
    ndays = year.valid_days
    wtd_yr = stacked_daily.strip.wtd[:ndays]       # (ndays, n)
    afp_yr = stacked_daily.strip.afp[:ndays]
    temp_yr = stacked_daily.T_soil_hydro[:ndays]   # (ndays, n_hydro)

    dfwt = pd.DataFrame(wtd_yr, index=pd.date_range(
        datetime.datetime(year.calendar_year, 1, 1), periods=ndays))
    dfafp = pd.DataFrame(afp_yr, ...)
    df_temp = pd.DataFrame(temp_yr, ...)

    # ── Annual biogeochemistry ──────────────────────────────────────

    # Heterotrophic respiration
    Rhet = heterotrophic_respiration_yr(df_temp, year.calendar_year, dfwt, ...)

    # Ground vegetation
    temp_sum = get_temp_sum(year.weather_yr)
    gv_state, gv_out = gvegetation.run_timestep(
        params.gvegetation, constants.gvegetation,
        gvegetation.assemble_inputs(
            ts=temp_sum,
            vol=state.annual_stand_out.volume,
            stems=state.annual_stand_out.stems,
            ba=state.annual_stand_out.basalarea,
            age=...
        ),
        state.annual_gv,
    )

    # Stand growth
    stand_inputs = stand.Inputs(
        photopara=..., forc=year.weather_yr, wt=dfwt, afp=dfafp,
        n_supply=np.zeros(n), p_supply=np.zeros(n), k_supply=np.zeros(n),
        groundvegetation_outputs=gv_out,
        previous_nut_stat=state.annual_stand_state.previous_nut_stat,
        calendar_year=year.calendar_year,
    )
    stand_state, stand_out, dom_out, sub_out, under_out = stand.grow_stand(
        state.annual_stand_state, constants.stand, stand_inputs,
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

    # ESOM: Mass, N, P, K
    inputs_mass = esom.assemble_inputs(
        tair_ts=..., tp_top_ts=df_temp.iloc[:, 2].values,
        tp_middle_ts=df_temp.iloc[:, 8].values,
        tp_bottom_ts=df_temp.iloc[:, 9].values,
        water_tables=dfwt.values,
        nonwoodylitter=...,
        woodylitter=...,
    )
    state_mass, yr_mass = esom.run_yr(
        params.esom.mass, constants.esom.mass, inputs_mass, state.annual_esom_mass,
    )
    # ... same for N, P, K ...

    doc_export = esom.compose_export(yr_mass, strip_diag, temp_yr, n)  # noqa

    # Methane
    _, ch4_out = methane.run_timestep(
        inputs=methane.assemble_inputs(year=year.calendar_year, dfwt=dfwt),
    )

    # Update nutrient status (nut_supply comes from ESOM + deposition + fert)
    nut_inputs = replace(stand_inputs,
        n_supply=yr_N.out_root_lyr + depoN + fert_out.nutrient_release["N"],
        p_supply=yr_P.out_root_lyr + depoP + fert_out.nutrient_release["P"],
        k_supply=yr_K.out_root_lyr + depoK + fert_out.nutrient_release["K"],
    )
    stand_state = stand.update_nutrient_status(stand_state, stand_out, nut_inputs)

    new_state = replace(state,
        annual_stand_state=stand_state,
        annual_stand_out=stand_out,
        annual_gv=gv_state,
        annual_esom_mass=state_mass,
        annual_esom_N=state_N,
        annual_esom_P=state_P,
        annual_esom_K=state_K,
    )
    return new_state, AnnualOutputs(
        daily=stacked_daily,
        stand=stand_out, stand_dom=dom_out, stand_sub=sub_out, stand_under=under_out,
        gv=gv_out,
        esom_mass=yr_mass, esom_N=yr_N, esom_P=yr_P, esom_K=yr_K,
        methane=ch4_out, fertilization=fert_out,
        Rhet=Rhet, soil_co2_balance=..., doc_export=doc_export,
    )
```

### run() — top-level orchestrator

```python
def run(simulation_params: SimulationParams) -> SimulationOutput:
    susi_params = simulation_params.susi_params
    weather = read_FMI_weather(...)      # (n_days, n_vars)

    params = _build_params(susi_params)
    constants = _compute_constants(params, susi_params, weather)

    # Pre-split forcings into (n_years, MAX_DAYS) — padding handled
    yearly_forcings = _build_yearly_forcings(susi_params, weather)

    init_state = _init_simulation_state(params, constants, susi_params, weather)

    final_state, all_annual = jax.lax.scan(
        _run_year_step, init_state, yearly_forcings,
    )

    _write_outputs(simulation_params, constants, all_annual)

    return SimulationOutput(annual=all_annual, final_state=final_state)
```

---

## JAX gradient example

```python
def simulate_and_loss(params, forcings):
    state = _init_simulation_state(params, ...)
    final, all_out = jax.lax.scan(_run_year_step, state, forcings)
    return jnp.sum(all_out.methane.ch4)

grad_fn = jax.grad(simulate_and_loss)
grads = grad_fn(params, forcings)
```

---

## Migration path

| Step | Changes | Testable |
|------|---------|----------|
| 1 | Enrich `strip.TimestepOutputs` (`wtd`, `ht`), `temperature.Outputs` (`T_soil_hydro`) | pytest |
| 2 | Extract `_run_daily_step` as pure function | pytest |
| 3 | Add `SimulationState`, `DailyOutputs`, `YearlyForcing`, `AnnualOutputs` types (no JAX) | pytest |
| 4 | Replace scenario loop with single pass (scenario dimension in writer only) | pytest |
| 5 | Build `_build_yearly_forcings`, pad days to `MAX_DAYS` | pytest |
| 6 | Inner `jax.lax.scan` over days in `_run_year_step` | pytest |
| 7 | Outer `jax.lax.scan` over years in `run()` | pytest + jax.grad |

---

## What changes in modules

| Module | Change | Reason |
|--------|--------|--------|
| `strip.py` | `TimestepOutputs` gains `wtd`, `ht` | Remove post-hoc computation from orchestrator |
| `strip.py` | `run_timestep` accepts `buffer=None` (allocates internally) | JAX compatibility (no mutable buffers) |
| `temperature.py` | `Outputs` gains `T_soil_hydro` | Remove post-hoc slicing from orchestrator |

No other module changes. Coupling quantities (`delta`, `total_runoff`, `surface_runoff`) live only in orchestrator types.

---
date created: Wednesday, January 10th 2024, 1:09:48 pm
date modified: Wednesday, June 3rd 2026, 8:45:25 am
---
See also: [[SuperSUSI rewrite working notes]]
# Overview

This document describes the architectural principles for the functional rewrite of the scientific model.

The goals are:

* Explicit dataflow
* Clear module boundaries
* JAX compatibility
* Easy testing and debugging
* Minimal hidden state
* Homogeneous module interfaces
* Scalable coupling between modules
* Long-term maintainability

The architecture follows a functional style rather than an object-oriented one.

---

# Core Principles

## 1. Explicit State

All evolving model state must be passed explicitly.

Modules should not:

* mutate global state
* own hidden mutable state
* directly modify other modules' states

Instead:

```python
new_state = run(state, static_inputs, params, dynamic_inputs)
```

The caller, `susi_main.py` owns orchestration.

---

## 2. Functional Style

Modules are implemented primarily as pure functions.

```python
def run(state, params, dynamic_inputs, static_inputs):
    ...
    return new_state 
```
---

## 3. Explicit Ownership

Each variable should ideally have:

* one owner
* one updater
* one authoritative definition

A module should only update its own state.

Coupling between modules should occur through exchanged fields or coupling variables.

---
# Terminology
- module: each piece of the interconnected dynamical system. Examples: hydrology module, ground vegetation module, ...
- main timestepping loop: is the core of the dynamical system simulation, where time passes and modules are coupled.
- orchestrator: the file that owns the main timestepping loop, in this case `susi_main.py`, and which conducts the couplings between modules (hence orchestra).

---

# Data Categories

The architecture strictly separates 4 categories of data.

They reflect differences in:
- mutability in loop: will its value change in the main simulation loop?
- ownership: who can change its value?
- pre-computability: can it be passed as an initial value to the simulation loop?

Here's a summary table, see below for a more detailed description of each.

| Category name       | mutability | owner                                   | pre-computability |
| ------------------- | ---------- | --------------------------------------- | ----------------- |
| `Params`            | ❌          | The "user" in the parameter input layer | yes               |
| `ComputedConstants` | ❌          | Orchestrator, before the main loop      | yes               |
| `Forcing`           | ✔️         | This module                             | yes               |
| `State`             | ✔️         | This module                             | no                |




## `Params`

These are constants, including scientific and algorithmic ones.
Most of them are received from a Pydantic model that is facing the user.
But it might be that we choose not to expose every single parameter to the user-facing input layer.
Those parameters should live in `Params`

They are set before calling `susi_main.run()`, but they are immutable afterwards.

They are generally independent from other `Params`, although we might allow some occasional trivial computations such as `length_cm = length_m/100`.

Examples: `dt`, `LAI`, `conductivity`.

## `ComputedConstants`

Inputs to the module that can be precomputed before the time stepping loop.
By definition, they cannot mutate inside the loop (or otherwise they could not be precomputed!)

They differ from `Params` in two ways:
- Conceptually, they are not parameters, but arrays, matrices, or other more complex structures.
- They may depend on other `Params` and other `ComputedConstants`

They may only depend on `Params` and `PrecomputedInput` from other modules.
```python
def compute_static_inputs(params: Params, other_static_inputs:PrecomputedInput)->PrecomputedInput:
```
The code for computing them lives in each module's .py file.

NOTE: `Params` and `ComputedConstants` are mutually exclusive: no constant lives in both simultaneously.

Examples:
- Constants that can be directly computed from some `Params` or other `PrecomputedInput`, such as `n_timesteps` if `dt` and `simulation_length` have been specified.
- Matrices, vectors needed for numerics, such as the matrix resulting from the discretization of a linear ODE.

## `Forcing`

Quantities that can be precomputed, but that change in the timestep.
Not owned by the module, but by weather.py, or whatever.

## `State`

Persistent evolving variables owned by the module. The variables that need to be "remembered" from the previous timestep.
Properties:
- They must be updated by the module during the timestep. Otherwise, they would belong in `ComputedConstants`.
- They must be used in the following timestep. If they are only used by other modules in the current timestep, then they should be `Outputs`.
- They may get written in the results netcdf file at the end of the computation, just like `Outputs`

## `Outputs`
Non-persistent evolving variables owned by the module. Properties:
- They are not needed in the next timestep. Otherwise, they would be part of `State`.
- They may be used by other modules downstream in the same timestep. For that reason, they might appear in the `Inputs` API (see below).


## Additional non-data category: Inputs.
`Inputs` are not a data category, but rather a bundle of other categories, which is convenient to describe the input API for each module. Note that the input API can have values from 
```python
def run_hydrology(params:Params, computed_constants:ComputedConstants, input: hydrology.Inputs)->hydrology.State:
```
In order to be explicit about what each module's inputs are and to be able to draw a relationship graph, it's best for each module to have a `assemble_inputs()` function as follows:
```python
def assemble_inputs(all_state: AllState, hydro_outputs: hydrology.Outputs, forcings: Forcings) -> Inputs:
    return Inputs(
        sst=all_state.ocean.sst,
        flux=a_outputs.flux,
        precip=forcings.precip,
    )
```
The only requirement for this to enable the Susi coupling graph is that there can be no conditional logic built inside `make_inputs()`.
This could be enforced by `ast` during CI.

--- 
# JAX compatibility

This data architecture is compatible with `jax.lax.scan`, and ensures that we can get all the benefits from it:
- We could bundle `Params` and `ComputedConstants` together into a single `Constants`, but then we'd lose the option of taking autodiff gradients over the parameters alone.
- `Forcings` need to be separated explicitly in `jax.lax.scan`.
- `State` and `Outputs` mirror `(carry, ys)` explicitly.


---

# Module Structure

All modules follow this architecture:

```python
@dataclass(frozen=True)
class Params:
    ...

@dataclass(frozen=True)
class ComputedConstants:
    ...

@dataclass(frozen=True)
class State:
    ...
	
@dataclass(frozen=True)
class Outputs:
    ...
	
@dataclass(frozen=True)
class Inputs:
    wtd : float
    rainfall : np.ndarray
    transpiration: float
	...
```

```python
def compute_initial_state(params: Params, constants: ComputedConstants, forcings:Forcings) -> State:
    ...


def make_inputs(all_state: AllState, hydro_outputs: hydrology.Outputs, forcings: Forcings) -> Inputs:
    return Inputs(
	...
    )

def run_timestep(
    params: Params,
    computed_constants: ComputedConstants,
    input: Inputs
) -> tuple[State, Outputs]:
    ...
```
---

# Main Timestep Loop

`susi_main.py` is responsible for: orchestration, coupling assembly, global scheduling.

Schematic view:
```python
params = AllParams(hydro_params, veg_params, etc.)
computed_constants = AllComputedConstants(...)


# Before time loop
hydro_state = hydro.compute_initial_state()

state = AllStates(hydro_state, etc.)

for year in years:
	# Inside time loop
	hydro_inputs = hydro.make_inputs(wtd=state.hydro.wtd, rainfall=forcing.rainfall, transpiration=state.vegetation.trans)
	hydro_state = hydro.run_timestep(
	    params=params.hydro,
	    constants=computed_constants.hydro,
	    inputs= hydro_inputs
	)
	
	# similar for vegetation and other modules
	
	new_state = AllStates(hydro_state, ...)
	output = AllOutput(...)
	
	return new_state, output
```

NOTE: careful here! AllState holds the state from the previous timestep, but `hydro_state` and `hydro_outputs` hold state from this timestep.

---
# JAX compatibility
The architecture is designed to work well with:

- Prefer frozen dataclasses:
```python
@dataclass(frozen=True)
```
- Dataclasses should be registered as JAX PyTrees.

Example:
```python
@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class State:
    ...
```
(Alternative libraries such as Equinox may also be appropriate.)

- The timestep loop should eventually be compatible with `lax.scan`:
```python
def step(state, forcing):
    ...
    return new_state

final_state, history = jax.lax.scan(
    step,
    initial_state,
    forcings,
)
```

This strongly favors:

* explicit state
* pure functions
* immutable data

---

# Functional Core, Imperative Shell

## Functional core

Contains:

* numerics
* timestepping
* kernels
* transformations

Must remain pure.

## Imperative shell

Contains:

* IO
* logging
* checkpointing
* visualization
* orchestration

This separation improves maintainability and testing.

---

# What `susi_main.py` should look like

The target architecture for the orchestrator resembles:
```python
state = initialize(params)

for forcing in forcings:

    ocean_inputs = ...
    atm_inputs = ...

    ocean_state= ocean.run(
        ocean_state,
        ocean_dynamic_inputs,
        ocean_params,
        ocean_static_inputs,
    )

    atm_state= atmosphere.run(
        atm_state,
        atm_dynamic_inputs,
        atm_params,
        atm_static_inputs,
    )
```

---
# Many different parameters!
The bio-geo-physical parameters to run SUSI are basically duplicated in two places:
- The `SusiParams` Pyantic model is user-facing and for validation. It is the surface of the model.
- `ModuleParams` contains the parameters grouped as used by each module. They are completely specified by `SusiParams`. 
The conversion between the two is done in `susi_main.py`:
```python
def _build_params(susi_params: SusiParams) -> ModuleParams:
```

The duplication is a constraint of `jax`, since it cannot take the Pydantic models directly.
But it is also a nice encapsulation.

---
# [experimental] Enforcing module structure
Define the expected interface once, centrally:

```python
# susi/module_api_protocol.py
from typing import Protocol, TypeVar

S = TypeVar("S")
P = TypeVar("P")
D = TypeVar("D")
I = TypeVar("I")

class SusiModule(Protocol[S, P, D, I]):
    def initialize(self, params: P) -> S: ...
    def run(self, state: S, dynamic_inputs: D, static_inputs: I, dt: float) -> S: ...
```
Each module's api.py then implicitly satisfies this protocol — no inheritance needed, structural subtyping handles it.
Then write a small test that explicitly checks conformance at import time:
```python
# tests/test_module_interfaces.py
from typing import get_type_hints
import importlib, pkgutil, susi.modules

def get_all_modules():
    return [
        importlib.import_module(f"susi.modules.{m.name}.api")
        for m in pkgutil.iter_modules(susi.modules.__path__)
    ]

def test_all_modules_have_run_and_initialize():
    for mod in get_all_modules():
        assert hasattr(mod, "run"), f"{mod.__name__} missing run()"
        assert hasattr(mod, "initialize"), f"{mod.__name__} missing initialize()"
        hints = get_type_hints(mod.run)
        assert "return" in hints, f"{mod.__name__}.run() missing return type annotation"
```
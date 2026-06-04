# Functional Rewrite Plan — Remaining Modules

Ranked from simplest to most difficult.

---

## 1. `stem_curve.py` (~287 lines) — **Stateless utility**

A class that groups pure math functions (stem taper curves, volume integration via `scipy.integrate.quad`, bucking). No mutable state.

**Difficulty:** Very low — just convert methods to standalone functions.

**Dependencies:** None (standalone).

**Required by:** `allometric_road_map.py` (calls `StemCurve().predictAssortmentVolumes()`).

**Called in `supersusi/core/susi_main.py`:** No. Not imported or called anywhere in the supersusi orchestrator.

---

## 3. `allometric_road_map.py` (~782 lines) — **Builder pattern**

`Growth_and_Yield_Table` stores parameters at init but all methods are effectively pure functions over DataFrames (`.get_table()`, `.stand_development()`, etc.).

**Difficulty:** Low — straightforward functionalization. Init data become function parameters.

**Dependencies:** `stem_curve` (⤴ #1), `weibull_recovery.py` (already functions), `metsi/` (already functions).

**Required by:** `susi_main.py` (indirectly, for pre-computing growth tables before simulation).

**Called in `supersusi/core/susi_main.py`:** No. Only used to prepare input data ahead of time (external to the main loop).

---

## 4. `esom.py` (~765 lines) — **Complex numerics, significant state**

The `Esom` class manages the M matrix (11-compartment OM decomposition), builds ~20 interpolation functions at init, and implements sparse-matrix-based decomposition with 9 rate constants. Has 4 substance variants (Mass, N, P, K).

**Difficulty:** Medium — algorithmically involved but self-contained.

**Dependencies:** `susi_utils.py` (already functions — `peat_hydrol_properties`, `wrc`).

**Required by:** `susi_main.py` (4 instances for Mass, N, P, K).

**Called in `supersusi/core/susi_main.py`:** Yes — lines 174–197 (instantiated as `Esom` OOP class), and used in the annual loop (lines 663–735, 737–748). Still OOP — needs full rewrite.

---

## 5. `canopylayer.py` (~981 lines) — **Most complex module**

Heavy mutable state (dozens of array attributes). Core algorithms: `assimilate()` (photosynthesis + leaf dynamics + biomass allocation + NPP), `leaf_dynamics()` (nutrient cycling with N/P/K), `cutting()` (thinning/logging).

**Difficulty:** High — tight coupling with `allometry`. Core of the forest growth model.

**Dependencies:** `allometry` (⤴ #2), `susi_utils.py` (already functions — `assimilation_yr`).

**Required by:** `stand.py` (encapsulates 3 Canopylayer instances for dominant/subdominant/under).

**Called in `supersusi/core/susi_main.py`:** No directly — accessed through `Stand` class methods (e.g., `stand.dominant.cutting()`, `stand.dominant.assimilate()`). Only indirectly via the `stand` instance.

---

## 6. `stand.py` (~727 lines) — **Layer orchestrator**

Orchestrates 3 `Canopylayer` instances. `assimilate()` sorts LAI by height and delegates to canopy layers; `update_nutrient_status()` uses Reineke's stand-density index.

**Difficulty:** Medium-high — must follow canopylayer rewrite. Primarily sums across layers but has real logic.

**Dependencies:** `canopylayer` (⤴ #5).

**Required by:** `susi_main.py` (the main orchestrator).

**Called in `supersusi/core/susi_main.py`:** Yes — extensively. Lines 124–134 (instantiation), 333 (reset_domain), 597–603 (assimilate + update), 619–624 (cutting), 737–748 (update_nutrient_status), and many attribute accesses throughout. Still OOP — critical rewrite target.

---

## 7. `susi_main.py` (~933 lines) — **Orchestrator, highest coupling**

The `Susi` class runs triple-nested loops (scenario × annual × daily) coordinating all modules.

**Difficulty:** High — must be done last since it imports everything. Not algorithmically hard, but requires all module APIs to be stable.

**Dependencies:** All of the above (⤴ #1–#6), plus all already-rewritten modules (strip, temperature, methane, mosslayer, gvegetation, canopygrid, fertilization).

**Should be done:** Last. The imperative shell that wires the functional core.

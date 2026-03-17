# **Implementation Plan: Two-Layer Folder Structure for SUSI Simulations**

**Objective:**
Introduce a consistent hierarchical folder layout for single and batch SUSI simulations while keeping metadata serializable and ensuring validation prevents conflicts and inconsistent configurations.

---

## **1. Folder Structure Design**

| Mode       | Folder Layout                             | Notes                                           |
| ---------- | ----------------------------------------- | ----------------------------------------------- |
| Single run | `experiment_id/`                          | Stand/scenario optional and unset.              |
| Batch run  | `experiment_id/stand_name/scenario_name/` | Both `stand_name` and `scenario_name` required. |


---

## **2. Metadata Model Updates** DONE

**SimulationMetaData changes:**

1. Add optional fields:

```python
stand_name: str | None = None
scenario_name: str | None = None
```

2. Update computed field for experiment folder path:

```python
@computed_field
@property
def experiment_folder_path(self) -> Path:
    base = self.parent_output_folder / self.experiment_id
    if self.stand_name and self.scenario_name:
        return base / self.stand_name / self.scenario_name
    return base
```

3. Validate that stand/scenario are either both set or both unset:

```python
@model_validator(mode="after")
def validate_stand_scenario_pair(self):
    if (self.stand_name is None) != (self.scenario_name is None):
        raise ValueError("stand_name and scenario_name must either both be set or both be None.")
    return self
```

---

## **3. MultipleSusis Model Updates** DONE

**New responsibilities:**

1. **Enforce batch structure consistency:**
   All runs define `stand_name`/`scenario_name`.

2. **Validate unique output paths:**
   Ensure no two runs point to the same folder.

3. Enforce all runs share the same `experiment_id`.

**Example validators:**

```python
def _check_consistent_run_structure(self):
    has_batch_structure = [
        run.metadata.stand_name is not None and run.metadata.scenario_name is not None
        for run in self.simulation_parameter_list
    ]
    if any(has_batch_structure) and not all(has_batch_structure):
        raise ValueError("Mixed single-run and batch-run structures detected.")
        
def _check_unique_output_paths(self):
    seen = set()
    for run in self.simulation_parameter_list:
        path = run.metadata.experiment_folder_path
        if path in seen:
            raise ValueError(f"Duplicate output path detected: {path}")
        seen.add(path)
```

---

## **4. Folder Creation Strategy** DONE

* Create folders **before simulations start**, outside the model layer:

```python
for run in multiple_simulations.simulation_parameter_list:
    path = run.metadata.experiment_folder_path
    path.mkdir(parents=True, exist_ok=False)
```

* This ensures no side effects occur during Pydantic validation.

---

## **5. Single vs. Batch Run Handling**

| Aspect                         | Single Run                 | Batch Run                                                     |
| ------------------------------ | -------------------------- | ------------------------------------------------------------- |
| `stand_name` / `scenario_name` | None                       | Required                                                      |
| Folder path computation        | `experiment_id/`           | `experiment_id/stand/scenario/`                               |
| Validation                     | Ensure path does not exist | Ensure all batch paths are unique and structure is consistent |
| Metadata serialization         | Unchanged                  | Unchanged, still stores `stand_name` and `scenario_name`      |

---

## **6. Validation Pipeline in MultipleSusis**

```python
@model_validator(mode="after")
def validate_configuration(self) -> "MultipleSusis":
    self._check_not_more_processes_than_runs()
    self._check_same_experiment_id()
    self._check_consistent_run_structure()
    self._check_for_duplicated_susi_params()
    self._check_for_duplicated_experiment_folder_paths()
    return self
```

**Notes:**

* `_check_not_more_processes_than_runs()` ensures at most one process per run.
* `_check_same_experiment_id()` prevents different roots in the same batch.
* `_check_consistent_run_structure()` enforces uniform single vs. batch configuration.
* `_check_for_duplicated_susi_params()` ensures no repeated simulation configurations.
* `_check_for_duplicated_experiment_folder_paths()` ensures no filesystem conflicts.

---

## **7. Implementation Steps: TDD**

1. **Metadata model:** DONE

   * Add `stand_name` and `scenario_name`.
   * Update `experiment_folder_path` computation.
   * Add validator to enforce pair consistency.
   * Validate folder-safe characters in `stand_name` and `scenario_name`.

2. **MultipleSusis model:** DONE

   * Add validators for consistent structure and unique paths.
   * Enforce same `experiment_id`.

3. **Folder creation utility:** DONE

   * Iterate through simulation list and create all output folders before running simulations.

4. **Testing:** DONE

   * Single-run mode: ensure paths are `experiment_id/` and validation passes.
   * Batch-run mode: test multiple stands/scenarios, ensure paths are unique and validation prevents duplicates.
   * Mixed single/batch in one `MultipleSusis`: should fail validation.

5. **Integration:** DONE

   * Update any existing simulation launch code to pass `stand_name` and `scenario_name` for batch runs.

---

## **8. Further Enhancements**



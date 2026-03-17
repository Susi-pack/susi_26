# Plan: Stand Age vs Allometry Validator

## Overview
Implement a validator in `SusiParams` that ensures stand ages for all layers (dominant, subdominant, under) fall within the valid age range defined in the corresponding allometry Excel files.

## Validation Rules
1. **Minimum age**: Initial stand age >= minimum age in the allometry file
2. **Maximum age**: Initial stand age + simulation duration <= maximum age in the allometry file

These rules apply to: `dominant`, `subdominant`, and `under` layers.

## Files Involved
- **Source**: `src/susi/io/susi_parameter_model.py` (lines 697-706)
- **Tests**: `tests/test_susi_model.py`
- **Test data**: `tests/data/test_allometry.xlsx` (copied from `src/inputs/SF_31.xlsx`)

## Implementation Steps

### Step 1: Copy allometry file (DONE)
- Copied `src/inputs/SF_31.xlsx` → `tests/data/test_allometry.xlsx`

### Step 2: Write tests (TDD - write failing tests first)

Create `tests/test_susi_model.py` with the following tests:

1. **`test_valid_stand_age_all_layers`**: Valid age within allometry range for all layers
   - Create SusiParams with initial ages in valid range
   - Expect: No validation error

2. **`test_initial_age_below_minimum`**: Initial age below minimum in allometry
   - Set initial_dominant_stand_age_years < min(allometry["age"])
   - Expect: `ValueError` with message about minimum age

3. **`test_initial_age_plus_duration_above_maximum`**: Age + simulation exceeds maximum
   - Set initial age such that initial + simulation_years > max(allometry["age"])
   - Expect: `ValueError` with message about maximum age

4. **`test_subdominant_layer_validation`**: Validate subdominant layer
   - Use subdominant allometry file, set invalid age
   - Expect: `ValueError`

5. **`test_under_layer_validation`**: Validate under layer
   - Use understorey allometry file, set invalid age
   - Expect: `ValueError`

**Helper fixture**: Create a minimal valid `SusiParams` object that can be used/modified across tests.

### Step 3: Implement validator

Complete the incomplete validator at `src/susi/io/susi_parameter_model.py:697-706`:

```python
@model_validator(mode="after")
def stand_age_vs_allometry_pathway(self) -> Self:
    """
    Validates the following:
    - Initial stand age is not below the minimum in the allometry file
    - initial stand age + simulation time is not above the maximum age in the allometry file
    """
    # 1. Calculate simulation duration in years
    simulation_duration_years = (
        self.simulation_config.end_date - self.simulation_config.start_date
    ).days / 365.25

    # 2. Validate dominant layer
    self._validate_layer_age(
        layer_name="dominant",
        initial_age=self.site_parameters.initial_dominant_stand_age_years,
        allometry_data=self.allometry_parameters.dominant_data,
        simulation_duration=simulation_duration_years,
    )

    # 3. Validate subdominant layer
    self._validate_layer_age(
        layer_name="subdominant",
        initial_age=self.site_parameters.initial_subdominant_stand_age_years,
        allometry_data=self.allometry_parameters.subdominant_data,
        simulation_duration=simulation_duration_years,
    )

    # 4. Validate under layer
    self._validate_layer_age(
        layer_name="under",
        initial_age=self.site_parameters.initial_understorey_age_years,
        allometry_data=self.allometry_parameters.under_data,
        simulation_duration=simulation_duration_years,
    )

    return self


def _validate_layer_age(
    self,
    layer_name: str,
    initial_age: float,
    allometry_data: dict[int, pd.DataFrame],
    simulation_duration: float,
) -> None:
    """Helper method to validate age for a single canopy layer."""
    # Skip if layer not in use (all keys are 0)
    if not allometry_data or all(k == 0 for k in allometry_data.keys()):
        return

    # Get first available allometry data
    df = next(iter(allometry_data.values()))
    min_age = df["age"].min()
    max_age = df["age"].max()

    if initial_age < min_age:
        raise ValueError(
            f"Initial {layer_name} stand age ({initial_age}) is below "
            f"minimum age ({min_age}) in allometry file"
        )

    if initial_age + simulation_duration > max_age:
        raise ValueError(
            f"Initial {layer_name} stand age ({initial_age}) plus simulation "
            f"duration ({simulation_duration:.1f} years) exceeds maximum age "
            f"({max_age}) in allometry file"
        )
```

### Step 4: Run tests
- Execute `pytest tests/test_susi_model.py`
- Verify all tests pass

## Notes
- The validator runs after all other validators (`mode="after"`)
- Simulation duration is calculated from `simulation_config.end_date - simulation_config.start_date`
- Each layer (dominant, subdominant, under) is validated independently
- If a layer is not in use (all keys = 0), validation is skipped for that layer

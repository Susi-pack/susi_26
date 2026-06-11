from supersusi.core import fertilization_types
from supersusi.core.fertilization_models import npk, ash, no_fertilization

ComputedConstants = ash.ComputedConstants | None
Params = ash.Params | npk.Params | no_fertilization.Params


def compute_constants(
    params: Params,
) -> ComputedConstants:
    match params:
        case ash.Params():
            return ash.compute_constants(params)
        case npk.Params():
            return None
        case no_fertilization.Params():
            return None


def assemble_inputs(params: Params, calendar_year: int) -> fertilization_types.Inputs:
    match params:
        case ash.Params() | npk.Params():
            # Temporal reference system translation:
            # If fertilization is started on the calendar year 2005,
            # and current calendar year is 2006,
            # this is the 1st year since fertilization started.
            years_since_fertilization = calendar_year - params.fpara.application_year

            return fertilization_types.Inputs(
                years_since_fertilization=years_since_fertilization,
            )
        case no_fertilization.Params():
            # Any value for years_since_fertilization will have no effect, due to no_fertilization.py
            return fertilization_types.Inputs(years_since_fertilization=0)


def run_timestep(
    params: Params,
    computed_constants: ComputedConstants,
    inputs: fertilization_types.Inputs,
) -> tuple[fertilization_types.State, fertilization_types.Outputs]:
    match params:
        case ash.Params():
            assert computed_constants is not None
            return ash.run_timestep(
                params=params,
                computed_constants=computed_constants,
                inputs=inputs,
            )
        case npk.Params():
            return npk.run_timestep(params=params, inputs=inputs)
        case no_fertilization.Params():
            # This returns zero fertilization effects
            return no_fertilization.run_timestep(params=params)

import netCDF4


def list_variable_absolute_paths(group: netCDF4.Dataset, path: str = "/") -> list[str]:
    vars_with_paths = []
    for v in group.variables:
        vars_with_paths.append(f"{path}/{v}".replace("//", "/"))

    for name, subgroup in group.groups.items():
        subpath = f"{path}/{name}".replace("//", "/")
        vars_with_paths.extend(list_variable_absolute_paths(subgroup, subpath))

    return vars_with_paths


def get_var_by_path(group: netCDF4.Dataset, var_path: str) -> dict:
    parts = var_path.strip("/").split("/")
    group = group
    for p in parts[:-1]:  # navigate to the group
        group = group.groups[p]
    return group.variables[parts[-1]][:]  # read the data

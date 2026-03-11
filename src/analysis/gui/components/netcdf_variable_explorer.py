import streamlit as st
from streamlit_tree_select import tree_select

from susi.io.load_output_data import NetcdfVariablePath, NetcdfVariableInfo


def build_tree_nodes(variables: dict[NetcdfVariablePath, NetcdfVariableInfo]):
    """Build tree-select nodes from variable paths.

    All node values must be unique, so we use full paths for both
    group nodes and leaf nodes.
    """
    # Build nested dictionary structure with full paths
    tree_dict = {}

    for var_path, var_info in variables.items():
        parts = [p for p in var_path.split("/") if p]
        current = tree_dict
        current_path = ""

        # Navigate/create path
        for i, part in enumerate(parts[:-1]):
            current_path = f"{current_path}/{part}" if current_path else f"/{part}"
            if part not in current:
                current[part] = {"children": {}, "full_path": current_path}
            current = current[part]["children"]

        # Add variable as leaf node
        var_name = parts[-1] if parts else var_info.name
        current[var_name] = {"variable": (var_path, var_info)}

    # Convert to tree-select format
    def dict_to_nodes(d, parent_path=""):
        nodes = []
        for key, value in d.items():
            if "variable" in value:
                # Leaf node (actual variable)
                var_info = value["variable"][1]
                var_path = value["variable"][0]
                nodes.append(
                    {
                        "label": f"{var_info.name}",
                        "value": var_path,
                        "title": f"Path: {var_path}\nShape: {var_info.shape}\nInfo: {var_info.units or 'N/A'}",
                    }
                )
            else:
                # Group node - use prefixed path to avoid conflicts with variables
                full_path = value.get("full_path", key)
                children = dict_to_nodes(value.get("children", {}), full_path)
                nodes.append(
                    {
                        "label": key,
                        "value": f"__group__{full_path}",  # Prefix to ensure uniqueness
                        "children": children,
                    }
                )
        return nodes

    return dict_to_nodes(tree_dict)


def build(
    netcdf_variables: dict[NetcdfVariablePath, NetcdfVariableInfo],
) -> dict[NetcdfVariablePath, NetcdfVariableInfo]:
    """
    Displays the netcdf variable tree and returns selected variables
    """
    col_left, col_right = st.columns(2)

    # Build tree and render
    tree_nodes = build_tree_nodes(netcdf_variables)

    with col_left:
        st.subheader("NetCDF Variable selection")
        # Use st.container with height parameter for scrollable area
        tree_container = st.container(height=400, border=True)
        with tree_container:
            tree_result = tree_select(
                tree_nodes,
                check_model="leaf",
                show_expand_all=True,
            )

    # Map selected paths back to NetcdfVariableInfo objects
    selected_paths = tree_result.get("checked", [])
    chosen_netcdf_variables = {
        var_path: var_info
        for var_path, var_info in netcdf_variables.items()
        if var_path in selected_paths
    }

    with col_right:
        st.subheader(f"Selected Variables ({len(chosen_netcdf_variables)})")
        for var_path, var_info in chosen_netcdf_variables.items():
            st.write(f"- `{var_path}` ({var_info.shape}, {var_info.units})")

    return chosen_netcdf_variables

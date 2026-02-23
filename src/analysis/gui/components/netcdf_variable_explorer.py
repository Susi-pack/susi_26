import streamlit as st
from streamlit_tree_select import tree_select

from susi.io.netcdf_utils import NetcdfVariableInfo


def build_tree_nodes(variables):
    """Build tree-select nodes from variable paths.

    All node values must be unique, so we use full paths for both
    group nodes and leaf nodes.
    """
    # Build nested dictionary structure with full paths
    tree_dict = {}

    for var in variables:
        parts = [p for p in var.path.split("/") if p]
        current = tree_dict
        current_path = ""

        # Navigate/create path
        for i, part in enumerate(parts[:-1]):
            current_path = f"{current_path}/{part}" if current_path else f"/{part}"
            if part not in current:
                current[part] = {"children": {}, "full_path": current_path}
            current = current[part]["children"]

        # Add variable as leaf node
        var_name = parts[-1] if parts else var.name
        current[var_name] = {"variable": var}

    # Convert to tree-select format
    def dict_to_nodes(d, parent_path=""):
        nodes = []
        for key, value in d.items():
            if "variable" in value:
                # Leaf node (actual variable)
                var = value["variable"]
                nodes.append(
                    {
                        "label": f"{var.name}",
                        "value": var.path,
                        "title": f"Path: {var.path}\nShape: {var.shape}\nInfo: {var.units or 'N/A'}",
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


def build(netcdf_variables: list[NetcdfVariableInfo]) -> list[NetcdfVariableInfo]:
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
    chosen_netcdf_variables = [
        var for var in netcdf_variables if var.path in selected_paths
    ]

    with col_right:
        st.subheader(f"Selected Variables ({len(chosen_netcdf_variables)})")
        for var in chosen_netcdf_variables:
            st.write(f"- `{var.path}` ({var.shape}, {var.units})")

    return chosen_netcdf_variables

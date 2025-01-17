from __future__ import annotations
import numpy as np
import pandas as pd
import yaml  # type: ignore

from fedeca.utils.fedeca_graphs.constants import DESCRIPTIONS_PATH
from fedeca.utils.fedeca_graphs.utils.data_classes import SharedState

CLASS_MAPPING = {
    "int": "int",
    "float": "float",
    "pandas.core.indexes.base.Index": "Index",
    "numpy.ndarray": "nparray",
    "numpy.float64": "float",
    "numpy.bool_": "bool",
    "NoneType": "NoneType",
    "pandas.core.series.Series": "Series",
    "dict": "dict",
    "set": "set",
    "list": "list",
    "bool": "bool",
    "pandas.core.frame.DataFrame": "DataFrame",
    "numpy.ma.core.MaskedArray": "MaskedArray",
    "str": "str",
    "numpy.int64" : "int",
}

PARAM_NOTATION = {
    "ngenes": "\\ngenes",
    "nnonzero_genes": "\\nzgenes",
    "nactive_genes": "\\nagenes",
    "Nls": "\\Nls",
    "Ngs": "\\Ngs",
    "nrefitted": "\\ngenesrefitted",
    "nparams": "\\nparams",
    "nlevelsk": "\\nlevels_k",
    "nlevels": "\\nlevels",
    "ntrendcoeffs": "2",
    "nadmissiblelevels": "|\\admissiblelevels|",
    "1": "1",
}

# With the aliases, we replace a shape value with a parameter.
# Each key in the dictionary is a shape value that needs to be replaced.
# The value is either a string or a tuple.
# If the value is a string, we replace the shape value with the string.
# If the value is a tuple, the first element is the default value to
#  replace the shape value.
# The second element is a dictionary with the key being the name of the item
# and the value being the value to replace the shape value.
# If this value is a string, we replace the shape value with the string.
# If this value is a tuple, the first element is the default value
# to replace the shape value.
# The second element is a dictionary with the key being the shared state number
# and the value being the value to replace the shape value.

ALIASES = {
    "64792": "ngenes",
    "57832": (
        "nnonzero_genes",
        {
            "local_features": "nactive_genes",
            "local_hat_matrix": (
                "nactive_genes",
                {64: "nnonzerogenes", 121: "nnonzerogenes", 123: "nnonzerogenes"},
            ),
            "local_nll": "nactive_genes",
            "irls_gene_names": "nactive_genes",
        },
    ),
    "1105": "nactive_genes",
    "1100": "nactive_genes",
    "244": "nactive_genes",
    "239": "nactive_genes",
    "20": "Nls",
    "100": "Ngs",
    "7910": (
        "nrefitted",
        {
            "local_features": "nactive_genes",
            "local_hat_matrix": "nactive_genes",
            "local_nll": "nactive_genes",
            "irls_gene_names": "nactive_genes",
        },
    ),
    "2": (
        "nparams",
        {
            "counts_by_lvl": "nlevels",
            "trend_coeffs": "ntrendcoeffs",
            "trimmed_mean_normed_counts": "nadmissiblelevels",
        },
    ),
    "1": ("1", {"unique_counts": "nlevelsk"}),
    "0": "nactive_genes",
}

TRIMMED_MEAN_IDS = {51, 52, 53, 54, 55, 58, 59, 60, 61, 62}
NAME_ALIASES = {"i_trimmed_mean": "$l \\in \\admissiblelevels$"}


def replace_alias(series: pd.Series) -> str:
    """
    Replace the shape value numbers with the corresponding parameters.

    Parameters
    ----------
    series : pd.Series
        The series containing the shape value, the name of the item and the old ID.

    Returns
    -------
    str
        The shape value with the corresponding parameter.
    """
    output_str = series["Shape"]
    for alias_dim in ALIASES:
        if alias_dim in output_str:
            output_str = output_str.replace(
                alias_dim,
                find_dimension_rpz(alias_dim, series["Name"], int(series["Old ID"])),
            )
    return output_str


def find_dimension_rpz(
    dimension_value: str, key_name: str, shared_state_no: int
) -> str:
    """
    Find the corresponding parameter for a dimension value.

    Parameters
    ----------
    dimension_value : str
        The dimension value to replace.
    key_name : str
        The name of the item.
    shared_state_no : int
        The shared state number.

    Returns
    -------
    str
        The corresponding parameter.

    Raises
    ------
    AssertionError
        If the shape info is not a tuple or string.
    """
    if dimension_value not in ALIASES:
        return dimension_value
    shape_info = ALIASES[dimension_value]
    if isinstance(shape_info, str):
        return PARAM_NOTATION.get(shape_info, shape_info)
    assert isinstance(shape_info, tuple)
    default_shape_info = shape_info[0]
    if key_name not in shape_info[1]:
        return PARAM_NOTATION.get(default_shape_info, default_shape_info)
    shape_name_info = shape_info[1][key_name]
    if isinstance(shape_name_info, str):
        return PARAM_NOTATION.get(shape_name_info, shape_name_info)
    assert isinstance(shape_name_info, tuple)
    if shared_state_no not in shape_name_info[1]:
        return PARAM_NOTATION.get(shape_name_info[0], shape_name_info[0])

    return PARAM_NOTATION.get(
        shape_name_info[1][shared_state_no], shape_name_info[1][shared_state_no]
    )


def extract_class_string(class_repr: str) -> str:
    """
    Extract the class string from a string representation of a class object.

    Parameters
    ----------
    class_repr : str
        The string representation of the class object.

    Returns
    -------
    str
        The class string without the <class ' and '> parts.
    """
    if class_repr.startswith("<class '") and class_repr.endswith("'>"):
        return CLASS_MAPPING[class_repr[8:-2]]
    return CLASS_MAPPING[class_repr]


def create_table_from_dico(
    shared_states: dict[int, SharedState],
    shared_with_aggregator: bool | None = None,
    shared_state_mapping: dict[int, int] | None = None,
    colsep: str = "0.75",
    parsize: str = "6",
) -> tuple[str, pd.DataFrame]:
    """
    Create a LaTeX table and a DataFrame from the shared states dictionary.

    This function generates a LaTeX table and a pandas DataFrame from a
    dictionary of shared states.
    The table can be filtered based on whether the states are shared
    with an aggregator, and the IDs
    can be updated using a provided mapping. The resulting table is
    formatted for LaTeX output with
    customizable column separation and paragraph size.

    Parameters
    ----------
    shared_states : dict[int, SharedState]
        A dictionary mapping IDs to SharedState objects. Each SharedState
        object contains information
        about the state, including its items, type, shape, and description.
    shared_with_aggregator : Optional[bool], optional
        If specified, filters the table to include only states that match
        the shared_with_aggregator field.
        By default, None, which includes all states regardless of this field.
    shared_state_mapping : dict[int, int] or None, optional
        A mapping from original shared state IDs to new IDs. If provided,
        the table will be restricted to
        the shared states in the mapping, and the IDs will be updated
        accordingly. By default, None.
    colsep : str, optional
        The column separation value for LaTeX formatting. By default, "0.75".
    parsize : str, optional
        The paragraph size for LaTeX formatting. By default, "6".

    Returns
    -------
    tuple[str, pd.DataFrame]
        A tuple containing:
        - A string representing the LaTeX table.
        - A pandas DataFrame containing the table data.

    Notes
    -----
    - The function reads descriptions from a YAML file specified by
      DESCRIPTIONS_PATH.
    - The table is sorted by the "ID" column.
    - Rows with old IDs in TRIMMED_MEAN_IDS and names "0" or "1" are
      removed and replaced with a
      template row indicating a trimmed mean.
    - The description column is updated based on the descriptions file.
    - The DataFrame is formatted for LaTeX output, with special handling
      for column names and shapes.
    - The LaTeX table is formatted as a longtable with customizable column
      separation and paragraph size.

    """
    with open(DESCRIPTIONS_PATH) as f:
        descriptions = yaml.safe_load(f)
    table = []
    for idx, shared_state in shared_states.items():
        if (
            shared_with_aggregator is not None
            and shared_state.shared_with_aggregator != shared_with_aggregator
        ):
            continue
        for name, item in shared_state.items.items():
            table.append(
                (
                    idx,
                    name,
                    item.type_str,
                    item.shape,
                    item.description,
                    shared_state.shared_with_aggregator,
                )
            )

    if shared_state_mapping is not None:
        # restrict to the shared states that are in the mapping
        table = [x for x in table if x[0] in shared_state_mapping]
        # update the IDs
        table = [(shared_state_mapping[x[0]], *x[:]) for x in table]  # type: ignore

    table.sort(key=lambda x: x[0])
    df = pd.DataFrame(
        table,
        columns=[
            "ID",
            "Old ID",
            "Name",
            "Type",
            "Shape",
            "Description",
            "Shared with aggregator",
        ],
    )

    # For trimmed mean related old ids, we want to remove the "0" and "1" Names
    # and instead replace them with "i for $0 \leq i \leq \nlevels-1$"

    # Remove rows with old ids in TRIMMED_MEAN_IDS and whose Name is "0" or "1"
    df = df[~((df["Old ID"].isin(TRIMMED_MEAN_IDS)) & (df["Name"].isin(["0", "1"])))]
    template_row = pd.Series(
        {
            "ID": 0,
            "Old ID": 0,
            "Name": "i_trimmed_mean",
            "Type": "dict",
            "Shape": "",
            "Description": "",
            "Shared with aggregator": False,
        }
    )
    new_rows = []
    for old_id in TRIMMED_MEAN_IDS:
        if old_id not in df["Old ID"].values:
            continue
        new_row = template_row.copy()
        new_row["Old ID"] = old_id
        # get the the first row of the df with the old id
        ref_row = df[df["Old ID"] == old_id].iloc[0]
        # Set the id of the new row as well as the shared with aggregator
        new_row["ID"] = ref_row["ID"]
        new_row["Shared with aggregator"] = ref_row["Shared with aggregator"]
        # Add the new row to the df
        new_rows.append(new_row)
    df = pd.concat([df, pd.DataFrame(new_rows)], axis=0)

    # resort the df by "ID"
    df = df.sort_values(by="ID")

    # Update the description column from the descriptions file
    # Create the lambda function to apply to descriptions.
    # the descriptions file is a dictionary with the name of the item as key
    # and as a value either a descirption or a dictioanry with the
    # old id as key and the description as value
    def get_description(series: pd.Series):
        if (item_name := series["Name"]) in descriptions:
            if isinstance(descriptions[item_name], dict):
                # Check if the old id is in the dictionary
                # Ids in this dict are strings "id1_id3" for example
                mapping_ids = {
                    int(k): v for v in descriptions[item_name] for k in v.split("_")
                }

                if (old_id := series["Old ID"]) in mapping_ids:
                    return descriptions[item_name][mapping_ids[old_id]]
            else:
                return descriptions[item_name]
        return pd.NA

    df["Description"] = df.apply(get_description, axis=1)

    if shared_with_aggregator is not None:
        df = df.drop(columns=["Shared with aggregator"])

    df["Type"] = df["Type"].apply(extract_class_string)

    df_to_save = df.copy().set_index(["ID", "Name"])
    # Replace names that have aliases
    df["Name"] = df["Name"].apply(lambda x: NAME_ALIASES.get(x, x))

    # Create a string from shape
    df["Shape"] = df["Shape"].apply(lambda x: str(x) if x is not None else "")

    df["Shape"] = df[["Shape", "Name", "Old ID"]].apply(
        lambda x: replace_alias(x), axis=1
    )

    # Latexify
    df["Shape"] = df["Shape"].apply(lambda x: f"${x}$" if len(x) > 0 else x)
    df["Name"] = df["Name"].apply(lambda x: x.replace("_", r"\_"))
    df["Type"] = df["Type"].apply(lambda x: x.replace("_", r"\_"))
    # Transform Shared with aggregator column to a Shared with column
    if shared_with_aggregator is None:
        df = df.rename(columns={"Shared with aggregator": "Shared with"})
        # map true to "Central server" and False to "Center" and nan to "All"
        df["Shared with"] = df["Shared with"].map(
            {True: "Server", False: "Center", pd.NA: "All", np.nan: "All", None: "All"}
        )

    # drop the old id column
    df = df.drop(columns=["Old ID"])

    df.set_index(["ID", "Name"], inplace=True)

    latex_table = df.to_latex()
    # replace \begin{tabular} with \begin{longtable}
    latex_table = latex_table.replace("tabular", "longtable")
    # Set the second to last column to be a ptype column
    latex_table = latex_table.replace(
        "llllll",
        (
            "l @{\\hspace{colsepfrac\\tabcolsep}}"
            "l@{\\hspace{colsepfrac\\tabcolsep}}l@{\\hspace{colsepfrac\\tabcolsep}}"
            "l@{\\hspace{colsepfrac\\tabcolsep}}"
            "p{parsizecm}@{\\hspace{colsepfrac\\tabcolsep}}l"
        )
        .replace("colsepfrac", colsep)
        .replace("parsize", parsize),
    )
    # Add \begin{tiny} and \end{tiny} around the table
    latex_table = "\\begin{tiny}\n" + latex_table + "\n\\end{tiny}"
    return latex_table, df_to_save

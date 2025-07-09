"""Plot file for timing experiments."""
from os.path import join

import pandas as pd
import seaborn as sns

from fedeca.utils.constants import EXPERIMENTS_PATHS
from fedeca.utils.experiment_utils import load_dataframe_from_pickles

# TODO use Owkin's palette
# from fedeca.viz.plot import owkin_palette


cmp = sns.color_palette("colorblind")
results = load_dataframe_from_pickles(
    EXPERIMENTS_PATHS["real_world"] / "results_Real-world_experiments.pkl"
)

fl_results = results.loc[results["backend_type"] != "N/A"]
# A bit ugly but the loop actually launched 10* pooled instead of 5
pooled_results = results.loc[results["backend_type"] == "N/A"]
pooled_results = pooled_results.groupby(["n_clients"]).head(5)

results = pd.concat([pooled_results, fl_results], ignore_index=True)


assert set(results["backend_type"].unique().tolist()) == set(["N/A", "simu", "remote"])

agg_df = results.groupby(["method", "backend_type", "n_clients"], as_index=False)[
    "fit_time"
].agg(["mean", "std", "count"])
assert all(agg_df["count"] == 5)


def e_to_latex(s):
    splitted = s.split("e")
    if len(splitted) == 2:
        before, exponent = splitted
        # get the sign of the exponent
        if exponent.startswith("+"):
            # remove the + sign and trim any leading zero
            exponent = exponent[1:].lstrip("0")
        elif exponent.startswith("-"):
            # keep the -1 sign, but trim any leading zero
            exponent = "-" + exponent[1:].lstrip("0")
        else:
            raise ValueError(f"Unexpected case {exponent}")
        if exponent == "":
            return before
        else:
            return before + "\cdot 10^{" + exponent + "}"  # noqa: W605
    else:
        return s


for col in ["mean", "std"]:
    agg_df[col] = agg_df[col].apply(lambda x: f"{x:.2e}")
    agg_df[col] = agg_df[col].apply(e_to_latex)

agg_df["mean"] = agg_df["mean"].apply(lambda x: "$" + x)
agg_df["std"] = agg_df["std"].apply(lambda x: x + "$")
#     agg_df[col] = pd.to_datetime(agg_df[col], unit="s").dt.strftime("%Hh %Mm %Ss.%f")
#     # trimming ms
#     agg_df[col] = [el[:-4] for el in agg_df[col].tolist()]
#     # removing useless hours
#     agg_df[col] = [
#         el[4:] if el.startswith("00h ") else el for el in agg_df[col].tolist()
#     ]
#     # remove useless minutes
#     agg_df[col] = [
#         el[4:] if el.startswith("00m ") else el for el in agg_df[col].tolist()
#     ]
#     # remove useless seconds
#     agg_df[col] = [
#         el[3:] if el.startswith("00s") else el for el in agg_df[col].tolist()
#     ]


agg_df["timing"] = (
    agg_df["mean"].astype(str).str.cat(agg_df["std"].astype(str), sep=r" \pm ")
)
agg_df = agg_df[["method", "backend_type", "n_clients", "timing"]]
agg_df = agg_df.rename(
    columns={
        "method": "Method",
        "backend_type": "Environment",
        "n_clients": "\#clients",  # noqa: E501, W605
        "timing": "Runtime (s)",
    }
)
agg_df["Environment"] = pd.Categorical(agg_df["Environment"]).rename_categories(
    {"remote": "real-world setup", "simu": "in-RAM"}
)
agg_df["Method"] = pd.Categorical(agg_df["Method"]).rename_categories(
    {"FedECA": "FedECA"}
)


print(agg_df)
print(agg_df.to_latex(index=False))

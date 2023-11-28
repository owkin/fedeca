# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def relative_error(x, y, absolute_error=False):
    if absolute_error:
        return np.abs(y - x) / np.abs(x)
    else:
        return np.linalg.norm(y - x) / np.linalg.norm(x)


cmp = sns.color_palette()
sns.set_palette(cmp)

methods = [
    "IPTW",
    "MAIC",
    # "CovAdj",
    "Naive",
]

results = pd.read_pickle("results_sim_vary_overlap.pkl")


# %%
results_grouped = results.groupby(["overlap", "exp_id"])


# Define a function to compute the relative error and ess for each group
def compute_error_ess(group):
    oracle_iptw = group[group["method"].str.lower() == "oracleiptw"]
    error_oracle = group.apply(
        lambda row: relative_error(
            oracle_iptw["exp(coef)"].values[0], row["exp(coef)"], absolute_error=True
        ),
        axis=1,
    )
    error_gt = group.apply(
        lambda row: relative_error(
            oracle_iptw["ate_true"].values[0], row["exp(coef)"], absolute_error=True
        ),
        axis=1,
    )

    df = pd.DataFrame(
        {
            "err_oracle": error_oracle,
            "err_ground_truth": error_gt,
            "method": group["method"],
            "exp_id": group["exp_id"],
            "overlap": group["overlap"],
            "ess": group["ess"],
        }
    )
    # Drop rows with 'oracleiptw' method
    df = df[df["method"].str.lower() != "oracleiptw"]

    return df


# Apply the function to each group
df_res = results_grouped.apply(compute_error_ess)

# Reset index
df_res = df_res.reset_index(drop=True)

# %%
fig, ax = plt.subplots()
sns.boxplot(x="overlap", y="err_oracle", hue="method", data=df_res)
ax.set_ylabel("Relative error of ATE (compared to Oracle IPTW)")
ax.set_yscale("log")
ax.set_xlabel("Degree of shift b/w arms (0=strong overlap)")
fig.tight_layout()

plt.savefig("test_overlap_error_oracle.png")
plt.show()
plt.clf()

# %%
fig, ax = plt.subplots()
sns.boxplot(x="overlap", y="err_ground_truth", hue="method", data=df_res)
ax.set_ylabel("Relative error of ATE (compared to ground truth)")
ax.set_yscale("log")
ax.set_xlabel("Degree of shift b/w arms (0=strong overlap)")
fig.tight_layout()

plt.savefig("test_overlap_error_gt.png")
plt.show()
plt.clf()

# %%
fig, ax = plt.subplots()
sns.boxplot(x="overlap", y="ess", hue="method", data=df_res)
ax.set_ylabel("ESS (for treated arm)")
ax.set_xlabel("Degree of shift b/w arms (0=strong overlap)")
fig.tight_layout()

plt.savefig("test_overlap_ess.png")
plt.show()
plt.clf()

# %%

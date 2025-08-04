"""Script for creating SMD and KM curves for ATE estimation."""

import argparse
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from fedeca.utils.constants import EXPERIMENTS_PATHS
from fedeca.viz.plot import owkin_palette, plot_survival_function

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble", action="store_true")
    args = parser.parse_args()

    with open(EXPERIMENTS_PATHS["pdac"] / "rw_pdac_smd.pkl", "rb") as f:
        df_smd = pickle.load(f)

    with open(EXPERIMENTS_PATHS["pdac"] / "rw_pdac_ate_km.pkl", "rb") as f:
        df_km = pickle.load(f)
        df_all = df_km[df_km["center"].eq("idibigi_ffcd_pancan")]
        df_ffcd_idibigi = df_km[df_km["center"].eq("idibigi_ffcd")]
        df_ffcd = df_km[df_km["center"].eq("ffcd")]
        df_idibigi = df_km[df_km["center"].eq("idibigi")]
        df_pancan = df_km[df_km["center"].eq("pancan")]

    if args.ensemble:
        _, ((ax_smd, ax_ffcd_idibigi), (ax_ffcd, ax_idibigi)) = plt.subplots(
            2, 2, figsize=(10.4, 7.8)
        )
        _, ((ax_all, ax_pancan)) = plt.subplots(1, 2, figsize=(10.4, 3.9))
    else:
        _, ax_smd = plt.subplots()
        _, ax_ffcd_idibigi = plt.subplots()
        _, ax_ffcd = plt.subplots()
        _, ax_idibigi = plt.subplots()
        _, ax_all = plt.subplots()
        _, ax_pancan = plt.subplots()

    for label, ax in zip("abcd", (ax_smd, ax_ffcd_idibigi, ax_ffcd, ax_idibigi)):
        ax.set_title(f"({label})", fontfamily="serif", loc="left")
    for label, ax in zip("ab", (ax_all, ax_pancan)):
        ax.set_title(f"({label})", fontfamily='serif', loc='left')

    sns.scatterplot(
        data=df_smd,
        x="smd",
        y="covariate",
        hue="type",
        palette=list(owkin_palette.values()),
        ax=ax_smd,
    )
    ax_smd.axvline(0.1, color="black", linestyle="--", alpha=0.2)
    ax_smd.axvline(-0.1, color="black", linestyle="--", alpha=0.2)
    ax_smd.set_xlabel("Standardized mean difference")
    ax_smd.set_ylabel("Baseline covariate")
    ax_smd.set_title("FFCD + IDIBGI")
    ax_smd.legend().set_title("")

    plot_survival_function(df_ffcd_idibigi, groupby="treatment", ax=ax_ffcd_idibigi)
    ax_ffcd_idibigi.set_title("FFCD + IDIBGI")

    plot_survival_function(df_ffcd, groupby="treatment", ax=ax_ffcd)
    ax_ffcd.set_title("FFCD")

    plot_survival_function(df_idibigi, groupby="treatment", ax=ax_idibigi)
    ax_idibigi.set_title("IDIBGI")

    plot_survival_function(df_all, groupby="treatment", ax=ax_all)
    ax_all.set_title("FFCD + IDIBGI + PanCAN")

    plot_survival_function(df_pancan, groupby="treatment", ax=ax_pancan)
    ax_pancan.set_title("PanCAN")

    for ax in (ax_ffcd_idibigi, ax_ffcd, ax_idibigi, ax_all, ax_pancan):
        ax.set(
            ylim=(0, 1),
            xlabel="Survival time (month)",
            ylabel="Probability of survival",
        )

    if args.ensemble:
        ax_smd.figure.tight_layout()
        ax_smd.figure.savefig("ffcd_idibigi_ensemble.pdf", bbox_inches="tight")
        ax_all.figure.tight_layout()
        ax_all.figure.savefig("km_all_pancan.pdf", bbox_inches="tight")
    else:
        ax_smd.figure.savefig("fed_smd_ffcd_idibigi.pdf", bbox_inches="tight")
        ax_ffcd_idibigi.figure.savefig("fed_km_ffcd_idibigi.pdf", bbox_inches="tight")
        ax_ffcd.figure.savefig("weighted_km_ffcd.pdf", bbox_inches="tight")
        ax_idibigi.figure.savefig("weighted_km_idibigi.pdf", bbox_inches="tight")
        ax_pancan.figure.savefig("weighted_km_pancan.pdf", bbox_inches="tight")
        ax_all.figure.savefig("fed_km_all.pdf", bbox_inches="tight")

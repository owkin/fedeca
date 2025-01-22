import matplotlib.pyplot as plt
from fedeca.utils.plot_fed_kaplan import compute_ci
import pickle
import re
import argparse
from pathlib import Path


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--results_name', "-R", type=str, help="Name of a file containing pairs of centers exchangeability experiments")

    args = parser.parse_args()

    results_name = args.results_name
   
    with open(Path(".") / results_name, "rb") as f:
        fl_results = pickle.load(f)

    names_mapping = {"ffcd": "FFCD", "idibigi": "IDIBGI", "pancan": "PanCAN"}
    TREATMENT = re.findall("(?<=treatment)[a-z]+", results_name)[0]

    POPULATION = re.findall("(?<=is_folfirinox)[a-zA-Z]+", results_name)[0] == "True"

    POPULATION = "FOLFIRINOX" if POPULATION else "GEM+NAB"

    color = "blue" if POPULATION == "FOLFIRINOX" else "orange"
    centers = set([name[1:-1] for name in Path(results_name).stem.split("_")[0][1:-1].split(", ")])
    #breakpoint()
    centers.remove(TREATMENT)
    CONTROL = list(centers)[0]
    TREATMENT = names_mapping[TREATMENT]
    CONTROL = names_mapping[CONTROL]
    fl_grid_treated, fl_s_treated, fl_var_s_treated, fl_cumsum_treated = fl_results[
        "treated"
    ]

    (
        fl_grid_untreated,
        fl_s_untreated,
        fl_var_s_untreated,
        fl_cumsum_untreated,
    ) = fl_results["untreated"]

    lower_untreated, upper_untreated = compute_ci(fl_s_untreated, fl_var_s_untreated, fl_cumsum_untreated)  # noqa: E501
    lower_treated, upper_treated = compute_ci(fl_s_treated, fl_var_s_treated, fl_cumsum_treated)


    ax = plt.plot(fl_grid_untreated, fl_s_untreated, label=CONTROL, color=color, linestyle='-')
    plt.fill_between(fl_grid_untreated, lower_untreated, upper_untreated, alpha=0.25, linewidth=1.0, step="post", color=color)  # noqa: E501

    ax = plt.plot(fl_grid_treated, fl_s_treated, label=TREATMENT, color=color, linestyle='--')
    plt.fill_between(fl_grid_treated, lower_treated, upper_treated, alpha=0.25, linewidth=1.0, step="post", color=color)  # noqa: E501

    plt.ylim(0., 1.)
    plt.ylabel("Probability of survival")
    plt.xlabel("Survival time (months)")
    plt.legend()
    plt.savefig(f"fed_km_T{TREATMENT}_C{CONTROL}_POP{POPULATION}.pdf", bbox_inches="tight", dpi=300)

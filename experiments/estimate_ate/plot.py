from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes


def plot_power(
    df_res: pd.DataFrame,
    fit_curve: bool = False,
    deg: int = 2,
    plot_kwargs: Optional[dict] = None,
) -> Axes:
    if plot_kwargs is None:
        plot_kwargs = {}
    plot_kwargs.setdefault("xlabel", "n_samples")
    axis = plot_kwargs.pop("ax", plt.gca())

    df_power = (
        df_res.groupby(["method", "n_samples"])
        .agg(
            power=pd.NamedAgg(column="p", aggfunc=lambda x: (x < 0.05).sum() / x.size),
        )
        .reset_index()
    )
    for name, group in df_power.groupby("method"):
        if (xlabel := plot_kwargs.pop("xlabel", None)) is not None:
            axis.set_xlabel(xlabel)
        if (ylabel := plot_kwargs.pop("ylabel", None)) is not None:
            axis.set_ylabel(ylabel)
        if fit_curve:
            fit = np.poly1d(np.polyfit(group["n_samples"], group["power"], deg=deg))
            axis.plot(group["n_samples"], fit(group["n_samples"]))
        axis.scatter("n_samples", "power", data=group, label=name, **plot_kwargs)
        axis.legend()

    return axis

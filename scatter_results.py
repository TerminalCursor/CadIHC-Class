#!/usr/bin/env python3
from lib.extensions.sqlite3_ext import legacy_sql_load as sql_load
from lib.tools.results_db import ResultsDatabase

SQLITE_FILE = "dump/Results.db"
results_df = sql_load(SQLITE_FILE, "RESULTS")

model_titles = {
    None: "All",
    "metrics": "Metrics",
    "energetics": "Energetics",
    "nuclei": "Metrics and Nuclei",
    "metricenergetics": "Metrics and Energetics",
    "latent": "Latent",
    "image": "CNN",
    "image_vt256": "VT-256",
    "image_vt512": "VT-512",
    "image_vt1024": "VT-1024",
}
class_titles = {
    "tn": "Triple Negative",
    "stage_basic": "Stage",
    # "stage_advanced": "Stage",
}
class_labels = {
    "tn": ["NTN", "TN"],
    "stage_basic": ["I", "II", "III"],
    # "stage_advanced": ["IA", "IIA", "IIB", "IIIA", "IIIB", "IIIC"],
}
scalar_metrics = ResultsDatabase.scalar_result_keys
matrix_metrics = ResultsDatabase.matrix_result_keys


def metric_1(df, x_var, y_var):
    """# A standard L2 metric but the distance from (1,1)"""
    df["XY"] = ((1 - df[[x_var, y_var]]) ** 2.0).sum(axis=1)
    max_row = df.loc[df["XY"] == df["XY"].min()].head(1)
    return max_row


def metric_2(df, x_var, y_var, d=2.0):
    """# Elliptical Distance
    Penalizes values off diagonal and so the lowest distances
    will be those that are close to the diagonal while remaining far from the origin

    d will be the squished-ness of the ellipse
    """
    from numpy import sin, cos, pi

    theta = pi / 4
    df["Xp"] = ((df[x_var] - 1) * cos(theta) + (df[y_var] - 1) * sin(theta)) / d
    df["Yp"] = (df[x_var] - 1) * sin(theta) - (df[y_var] - 1) * cos(theta)
    df["XY"] = (df[["Xp", "Yp"]] ** 2.0).sum(axis=1)
    max_row = df.loc[df["XY"] == df["XY"].min()].head(1)
    return max_row


def metric_3(df, x_var, y_var):
    """# The Hyperbolic Distance Metric"""
    df["XY"] = df[x_var] * df[y_var] ** 0.5
    max_row = df.loc[df["XY"] == df["XY"].max()].head(1)
    return max_row


def run(x_var, y_var, category, results_df, metric_info):
    from matplotlib import pyplot as plt

    metric = metric_info["fptr"]
    metric_name = metric_info["name"]

    plt.clf()
    plot_kwargs = {
        "cmap": "hsv",
        "s": 16,
        "zorder": 0,
    }

    # Do any filtering here
    # results_df = results_df.filter_on("NAME", "latent")
    results_df = results_df.filter_on("CATEGORY", category)

    # New columns
    results_df["MODEL"] = results_df["NAME"].map(results_df["NAME"].enumerate())

    M = results_df.get("NAME")
    models = M.unique()

    plot_kwargs["vmin"] = 0
    plot_kwargs["vmax"] = len(models)

    # Plot Metric Scatter
    for m_idx, model in enumerate(models):
        df = results_df.filter_on("NAME", model)
        max_row = metric(df, x_var, y_var)
        x, y, d = max_row[[x_var, y_var, "XY"]].to_numpy().flatten()
        print(f"{model:<16}: {d}")
        X = df.get(x_var)
        Y = df.get(y_var)
        plt.scatter(X, Y, c=df.get("MODEL"), label=model_titles[model], **plot_kwargs)
        # Star Best Performers
        cm = plt.get_cmap("hsv")
        plt.scatter(
            x,
            y,
            marker="*",
            edgecolors="black",
            linewidths=0.5,
            color=cm(m_idx / len(models)),
            zorder=2,
        )

    # Draw Line of random chance
    _N = len(class_labels[category])
    plt.plot(
        [1 / _N, 1 / _N, -0.1, -0.1, 1],
        [1, -0.1, -0.1, 1 / _N, 1 / _N],
        "r--",
        label="Random Choice",
        zorder=-1,
    )

    plt.legend(loc="upper left", fontsize=8)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.gca().set_aspect("equal")
    _XLAB = x_var.capitalize().replace("_", " ").replace(" c", " (Consensus)")
    plt.xlabel(_XLAB)
    _YLAB = y_var.capitalize().replace("_", " ").replace(" c", " (Consensus)")
    plt.ylabel(_YLAB)
    plt.title(class_titles[category])
    plt.savefig(f"Outputs/{category}_{metric_name}_{x_var}_{y_var}.png", dpi=600)


categories = class_titles.keys()

from itertools import product, combinations

metric_options = [
    {
        "name": "CL2Q",
        "fptr": metric_1,
    },
    {
        "name": "EL2Q",
        "fptr": metric_2,
    },
    {
        "name": "HQ",
        "fptr": metric_3,
    },
]

for (x_var, y_var), category, metric_info in product(
    combinations(scalar_metrics, 2), categories, metric_options
):
    run(x_var, y_var, category, results_df, metric_info)

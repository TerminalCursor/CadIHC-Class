#!/usr/bin/env python3
from histclass.extensions.sqlite3_ext import legacy_sql_load as sql_load
from histclass.tools.results_db import ResultsDatabase

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
    # max_row = df.loc[df["XY"] == df["XY"].min()].copy()
    max_row = df.copy().sort_values(by="XY", ascending=True)
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
    max_row = df.loc[df["XY"] == df["XY"].min()].copy()
    return max_row


def metric_3(df, x_var, y_var):
    """# The Hyperbolic Distance Metric"""
    df["XY"] = df[x_var] * df[y_var] ** 0.5
    max_row = df.loc[df["XY"] == df["XY"].max()].copy()
    return max_row


def run(x_var, y_var, category, results_df, metric_info):
    from matplotlib import pyplot as plt
    from pandas import concat

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
    rows = []
    for m_idx, model in enumerate(models):
        df = results_df.filter_on("NAME", model)
        max_row = metric(df, x_var, y_var)
        max_row = max_row.head(3)
        rows.append(max_row)
    print(concat(rows))


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

scalar_metrics = ["validation_accuracy", "training_accuracy"]
scalar_metrics = ["validation_accuracy", "validation_accuracy_c"]
metric_options = [metric_options[0]]

for (x_var, y_var), category, metric_info in product(
    combinations(scalar_metrics, 2), categories, metric_options
):
    print(f"({x_var}, {y_var}): {category}, {metric_info}")
    run(x_var.upper(), y_var.upper(), category, results_df, metric_info)

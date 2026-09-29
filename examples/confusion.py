#!/usr/bin/env python3
from histclass.extensions.sqlite3_ext import legacy_sql_load as sql_load
from histclass.tools.results_db import ResultsDatabase
from numpy import array, diag
from matplotlib import pyplot as plt

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


def run(class_type, absolute, column_normalized, matrix_variable, results_df):
    if absolute and not column_normalized:
        return

    # Do any filtering here
    results_df = results_df.filter_on("CATEGORY", class_type)

    models = list(model_titles.keys())

    for model in models:
        df = results_df.copy()
        if model is not None:
            df = df.filter_on("NAME", model)
        # Evaluate confusion matrices
        series = df[matrix_variable].to_numpy()
        matrices = array([eval(f"array{v}") for v in series])

        averaged = matrices.mean(axis=0)
        if column_normalized:
            col_norm = averaged / averaged.sum(axis=0)
        else:
            col_norm = averaged
        accuracy = diag(averaged).sum() / averaged.sum()
        print(model_titles[model])
        print(accuracy)

        # Plotting
        plt.clf()
        plt.imshow(col_norm, vmin=0, vmax=1 if absolute else col_norm.max())
        plt.xlabel("Actual Class")
        plt.ylabel("Predicted Class")
        plt.xticks(range(len(class_labels[class_type])), class_labels[class_type])
        plt.yticks(range(len(class_labels[class_type])), class_labels[class_type])
        plt.title(model_titles[model])
        plt.colorbar()
        IS_ABS = "ABS" if absolute else "REL"
        IS_COL = "COL" if column_normalized else "RAW"
        model_name = model_titles[model].replace(" and ", "")
        plt.savefig(
            f"Outputs/{IS_ABS}_{IS_COL}_{class_type}_{matrix_variable}_{model_name}.png",
            dpi=600,
        )


from itertools import product

for class_type, absolute, column_normalized, matrix_variable in product(
    class_labels.keys(), [False, True], [False, True], matrix_metrics
):
    run(class_type, absolute, column_normalized, matrix_variable, results_df)

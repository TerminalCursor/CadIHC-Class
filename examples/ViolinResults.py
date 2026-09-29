from pickle import load
import pandas as pd
import sqlite3
from matplotlib import pyplot as plt
from json import dumps
import numpy as np
from itertools import combinations
from itertools import product
plt.rc('axes', linewidth=4)

BASE = "FigurePieces/FigureX-"


def violin(x, loc, color1, color2, w=0.15, alpha=1, a2=0.75):
    viol = plt.violinplot(
        x,
        positions=[
            loc,
        ],
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for pc in viol["bodies"]:
        pc.set_facecolor(color1)
        pc.set_edgecolor(color2)
        pc.set_alpha(alpha)
    plt.plot([loc - w, loc + w], [min(x), min(x)], c=color2, alpha=a2)
    plt.plot([loc, loc], [min(x), max(x)], c=color2, alpha=a2)
    plt.plot([loc - w, loc + w], [max(x), max(x)], c=color2, alpha=a2)


def jitter(v, noise_level=5e-2):
    return v + noise_level * (2 * np.random.rand() - 1)


def _round(v: str) -> str:
    rnd = int(float(v) * 1000)
    rnd /= 10
    rnd = int(round(rnd))
    return f"{rnd}"

odf = None
with sqlite3.connect("Results.db") as connection:
    odf = pd.read_sql_query("SELECT * FROM RESULTS", connection)
#odf = odf[(odf["NAME"] == "image_vt-attention2hilostage") | (odf["NAME"] == "image_vt-attention2hilograde")]
odf = odf[(odf["NAME"] == "image_vt-attention2hilostage") | (odf["NAME"] == "image_vt-attention2hilograde")]

ddf = None
with sqlite3.connect("dump.debug/Results.db") as connection:
    ddf = pd.read_sql_query("SELECT * FROM RESULTS", connection)
ddf = ddf[(ddf["CATEGORY"] == "tn")]

cdf = None
with sqlite3.connect("dump.earlate/Results.db") as connection:
    cdf = pd.read_sql_query("SELECT * FROM RESULTS", connection)
cdf = cdf[(cdf["CATEGORY"] == "tn")]

bdf = None
with sqlite3.connect("dump.mne/Results.db") as connection:
    bdf = pd.read_sql_query("SELECT * FROM RESULTS", connection)

df = None
with sqlite3.connect("dump.imarm/Results.db") as connection:
    df = pd.read_sql_query("SELECT * FROM RESULTS", connection)
assert df is not None
if odf is not None:
    df = pd.concat([df,odf], ignore_index=True)
if bdf is not None:
    df = pd.concat([df,bdf], ignore_index=True)

print(df.keys())
print(df.head())
print(np.unique(df["NAME"]))

# Data structure:
#  (1, 9) for consensus
#  ("t", "v") for split
#  ("tp", "grade", "stage_basic") for category
#  (""...) for model name
#  Metric name ("total_accuracy")
# print(list(data[1]["v"]["tn"]["image"].keys()))

cons_map = {
    "": ("One-Shot", ("red", "darkred")),
    "_C": ("Consensus", ("lightblue", "blue")),
}

model_map = {
    "image": "CNN",
    "image_vt": "ViT",
    "energetics": "Energy",
    "morphology": "Morphology",
    "morphologynuclei": "Morphology \\& Nuclei",
    "morphologyenergeticsnuclei": "Morphology, Nuclei, \\& Energetics",
    "nuclei": "Nuclei",
    "morphologyenergetics": "Morphology \\& Energy",
    "latent": "Latent Embedding",
}

groupings = [
    ("image", "image_vt"),
    ##("energetics",),
    ##("morphology", "nuclei", "morphologynuclei", "morphologyenergetics", "latent"),
    #### ("energetics","morphologyenergetics", "morphologyenergeticsnuclei"),
    ("morphology", "nuclei", "morphologynuclei"),
    # ("morphology", "morphologynuclei", "energetics"),
]

categories = [
    # "tn",
    "hilostage",
    "hilograde",
]

skey = "VALIDATION_ACCURACY"
mkey = "total_accuracy"

# Format:
# For each of the groupings: (CNN, ViT), (Energetics), (Remaining PC)
# Plot the metrics with the one shot accuracy in one color and conensus in another
# Plot (TN, Grade, Stage) in different plots
# Add scatter of points
full_data = df.copy()
full_data["DIST"] = (
    (1 - full_data["TRAINING_ACCURACY_C"]) ** 2.0
    + (1 - full_data["TRAINING_ACCURACY"]) ** 2.0
    + (1 - full_data["VALIDATION_ACCURACY_C"]) ** 2.0
    + (1 - full_data["VALIDATION_ACCURACY"]) ** 2.0
)
#test = full_data.copy()
#print(test[test["TIMESTAMP"] == '20260514-154819-983493'])
#test = test[test["CATEGORY"] != "tn"].reset_index(drop=True)
#test = test[test["CATEGORY"] == "hilograde"].reset_index(drop=True)
#pd.set_option('display.max_rows', None)
#test_idx = test["DIST"].argsort().to_numpy()
#print(test.loc[test_idx])
for grouping in groupings:
    for cidx, category in enumerate(categories):
        plt.clf()
        x_loc = 0
        c = 0.5
        xticks = []
        ticks = []
        for midx, model in enumerate(grouping):
            for cons in ["", "_C"]:
                print(category, model)
                cons_name, cons_colors = cons_map[cons]
                scale = 3
                offset = 1
                if len(grouping) == 1:
                    offset = 0.75
                elif len(grouping) == 2:
                    scale = 2.4
                x_loc = midx * scale
                if cons == "_C":
                    x_loc += offset
                data = full_data.copy()
                data = data[
                    (data["CATEGORY"] == category)
                    & (data["NAME"] == model + "-attention2" + category)
                ]
                indicies = data["DIST"].argsort().to_numpy()[:10]
                data = data.iloc[indicies]
                vdata = data[
                    (data["CATEGORY"] == category)
                    & (data["NAME"] == model + "-attention2" + category)
                ][skey + cons].to_numpy()
                ticks.append(model)
                xticks.append(x_loc)
                violin(vdata, x_loc, *cons_colors, alpha=0.5)
                continue
                for vd in vdata:
                    plt.scatter(
                        jitter(x_loc), vd, s=64, color=cons_colors[1], alpha=0.25
                    )
        plt.plot([0 - 0.25, x_loc + 0.25], [c, c], "r--")
        plt.ylim(0, 1)
        plt.yticks([0, 0.5, 1], [])

        plt.xticks(xticks, ticks, rotation=90)
        plt.tight_layout()
        plt.savefig(f"{BASE}_{category}_{''.join(grouping)}.png")
        plt.xticks(xticks, [])
        plt.tight_layout()
        plt.savefig(f"{BASE}_{category}_{''.join(grouping)}.svg")

models = [v for g in groupings for v in g]
for model in models:
    print(f"{model_map[model]:40}", end="")
    for category in categories:
        data = full_data.copy()
        data = data[
            (data["CATEGORY"] == category)
            & (data["NAME"] == model + "-attention2" + category)
        ]
        indicies = data["DIST"].argsort().to_numpy()[:10]
        data = data.iloc[indicies]
        vdata = data[(data["NAME"] == model + "-attention2" + category)]
        os = vdata[skey].max()
        os_sd = vdata[skey].std()
        cons = vdata[skey + "_C"].max()
        cons_sd = vdata[skey + "_C"].std()
        os = int(os * 100)
        os_sd = int(os_sd * 100)
        cons = int(cons * 100)
        cons_sd = int(cons_sd * 100)
        print(f" & {os} \\pm {os_sd:2d} \\% & {cons} \\pm {cons_sd:2d} \\%", end="")
    print(" \\\\")

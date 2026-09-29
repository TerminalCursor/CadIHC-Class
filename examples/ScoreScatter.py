from pickle import load
from matplotlib import pyplot as plt
from json import dumps
import numpy as np
from itertools import combinations
from itertools import product


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
    plt.plot([loc-w,loc+w], [min(x),min(x)], c=color2, alpha=a2)
    plt.plot([loc,loc], [min(x), max(x)], c=color2, alpha=a2)
    plt.plot([loc-w,loc+w], [max(x),max(x)], c=color2, alpha=a2)


def jitter(v, noise_level=5e-2):
    return v + noise_level * (2 * np.random.rand() - 1)


def _round(v: str) -> str:
    rnd = int(float(v) * 1000)
    rnd /= 10
    rnd = int(round(rnd))
    return f"{rnd}"


data = {}
tdata = {}

with open("outputs/scores.pickle.orig", "rb") as fhandle:
    data = load(fhandle)

# Data structure:
#  (1, 9) for consensus
#  ("t", "v") for split
#  ("tp", "grade", "stage_basic") for category
#  (""...) for model name
#  Metric name ("total_accuracy")
# print(list(data[1]["v"]["tn"]["image"].keys()))

cons_map = {
    1: ("One-Shot", ("red", "darkred")),
    9: ("Consensus", ("lightblue", "blue")),
}

groupings = [
    ("image", "image_vt"),
    ("energetics",),
    ("metrics", "nuclei_only", "nuclei", "metricenergetics", "latent"),
]

categories = [
    "tn",
    "grade",
    "stage_basic",
]

skey = "v"
mkey = "total_accuracy"

# Format:
# For each of the groupings: (CNN, ViT), (Energetics), (Remaining PC)
# Plot the metrics with the one shot accuracy in one color and conensus in another
# Plot (TN, Grade, Stage) in different plots
# Add scatter of points
for grouping in groupings:
    for cidx, category in enumerate(categories):
        plt.clf()
        x_loc = 0
        c = 1./3.
        if category == "tn":
            c = 0.5
        for midx, model in enumerate(grouping):
            for cons in [1, 9]:
                cons_name, cons_colors = cons_map[cons]
                scale = 3
                offset = 1
                if len(grouping) == 1:
                    offset = 0.75
                elif len(grouping) == 2:
                    scale = 2.4
                x_loc = midx * scale
                if cons == 9:
                    x_loc += offset
                vdata = data[cons][skey][category][model][mkey]
                violin(vdata, x_loc, *cons_colors, alpha=0.5)
                # plt.violinplot(vdata, positions=[x_loc,], color=cons_color)
                for vd in vdata:
                    pass
                    plt.scatter(jitter(x_loc), vd, s=64, color=cons_colors[1], alpha=0.25)
        plt.plot([0-0.25,x_loc+0.25],[c,c], 'r--')
        plt.ylim(0, 1)
        plt.xticks([], [])
        plt.yticks([0,0.33,0.5,1], [])
        plt.tight_layout()
        plt.savefig(f"outputs/new_figs/ta_{''.join(grouping)}_{category}.png")
        plt.savefig(f"outputs/new_figs/ta_{''.join(grouping)}_{category}.svg")

from pickle import load
from matplotlib import pyplot as plt
import numpy as np
from itertools import combinations
from itertools import product

data = {}
with open("outputs/scores.pickle", "rb") as fhandle:
    data = load(fhandle)

tv = "t"
category_labels = {
    "tn": ["Non-Triple Negative", "Triple Negative"],
    "stage_basic": ["I", "II", "III"],
}
category = "tn"
cat_idx = 0


for tv, category in product(["t", "v"], list(category_labels.keys())):
    for cat_idx in range(len(category_labels[category])):
        models = list(data[tv][category].keys())
        plt.clf()
        for index, name in enumerate(models):
            _data = data[tv][category][name]["score"]
            tpr = np.array(_data["ntpr"])[:, cat_idx]
            tnr = np.array(_data["ntnr"])[:, cat_idx]
            # tpr = np.array(_data['tpr'])[:,cat_idx]
            # tnr = np.array(_data['tnr'])[:,cat_idx]
            plt.scatter(tpr, tnr, label=name)
        plt.xlabel("tpr")
        plt.ylabel("tnr")
        plt.title(category_labels[category][cat_idx])
        plt.legend()
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.savefig(f"outputs/scores_{tv}_{category}_{cat_idx}.png")
        plt.close()

for tv, category in product(["t", "v"], list(category_labels.keys())):
    for cat_idx in range(len(category_labels[category])):
        models = list(data[tv][category].keys())
        plt.clf()
        box_data = [[] for _ in range(len(models))]
        for index, name in enumerate(models):
            _data = data[tv][category][name]["score"]
            aucs = np.array(_data["auc"])[:, cat_idx]
            for auc in aucs.flatten():
                box_data[index].append(auc)
        #       print(box_data)
        plt.boxplot(box_data, patch_artist=True, tick_labels=models)
        plt.plot([1, len(models)], [0.5, 0.5], "r--")
        plt.xlabel("model")
        plt.ylabel("auc")
        plt.title(category_labels[category][cat_idx])
        plt.ylim([0, 1])
        plt.savefig(f"outputs/box_{tv}_{category}_{cat_idx}.png")
        plt.close()

for category in category_labels.keys():
    for cat_idx in range(len(category_labels[category])):
        models = list(data[tv][category].keys())
        plt.clf()
        box_data = [[] for _ in range(len(models))]
        for index, name in enumerate(models):
            t_data = data["t"][category][name]["score"]
            v_data = data["v"][category][name]["score"]
            aucs_t = np.array(t_data["auc"])[:, cat_idx]
            aucs_v = np.array(v_data["auc"])[:, cat_idx]
            plt.scatter(aucs_t, aucs_v, label=name)
        plt.legend()
        # plt.plot([0,1],[0,1], 'r--')
        plt.xlabel("auc training")
        plt.ylabel("auc validation")
        plt.title(category_labels[category][cat_idx])
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.savefig(f"outputs/auc_{category}_{cat_idx}.png")
        plt.close()

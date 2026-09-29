from pickle import load
from matplotlib import pyplot as plt
import numpy as np
from itertools import combinations
from itertools import product

data = {}
with open("outputs/scores.pickle", "rb") as fhandle:
    data = load(fhandle)
#print(data)

tv = "t"
category_labels = {
    "tn": ["Non-Triple Negative", "Triple Negative"],
    "stage_basic": ["I", "II", "III"],
    "grade": ["1", "2", "3"],
}
category = "tn"
cat_idx = 0
consensi = [1,9]
consensus = 9
usages = ["t"]
usage = "v"

data = data[consensus]

for key in category_labels.keys():
    if key not in data["t"].keys():
        data["t"][key] = {}
        data["v"][key] = {}


for tv, category in product(usages, list(category_labels.keys())):
    for cat_idx in range(len(category_labels[category])):
        models = list(data[tv][category].keys())
        if len(models) == 0:
            continue
        plt.clf()
        for index, name in enumerate(models):
            _data = data[tv][category][name]["score"]
            tpr = np.array(_data["tpr"])[:, cat_idx]
            tnr = np.array(_data["tnr"])[:, cat_idx]
            plt.scatter(tpr, tnr, label=name)
        plt.xlabel("tpr")
        plt.ylabel("tnr")
        plt.title(category_labels[category][cat_idx])
        plt.legend()
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.savefig(f"outputs/scores_{tv}_{category}_{cat_idx}_{consensus}.png")
        plt.close()

for tv, category in product(usages, list(category_labels.keys())):
    models = list(data[tv][category].keys())
    if len(models) == 0:
        continue
    plt.clf()
    box_data = [[] for _ in range(len(models))]
    for index, name in enumerate(models):
        _data = data[tv][category][name]["total_accuracy"]
        aucs = np.array(_data)[:]
        for auc in aucs.flatten():
            box_data[index].append(auc)
    #       print(box_data)
    #plt.boxplot(box_data, patch_artist=True, tick_labels=models)
    plt.violinplot(box_data)
    plt.xticks(range(1,len(models)+1), models, rotation=45)
    plt.plot([1-0.5, len(models)+0.5], [1/len(category_labels[category]), 1/len(category_labels[category])], "r--")
    plt.xlabel("model")
    plt.ylabel(f"Total Accuracy")
    plt.title(category)
    plt.ylim([0, 1])
    plt.savefig(f"outputs/box_tot_acc_{tv}_{category}_{consensus}.png")
    plt.close()

for score in ["auc", "acc", "ef1", "tpr", "tnr"]:
    for tv, category in product(usages, list(category_labels.keys())):
        for cat_idx in range(len(category_labels[category])):
            models = list(data[tv][category].keys())
            if len(models) == 0:
                continue
            plt.clf()
            box_data = [[] for _ in range(len(models))]
            for index, name in enumerate(models):
                _data = data[tv][category][name]["score"]
                aucs = np.array(_data[score])[:, cat_idx]
                for auc in aucs.flatten():
                    box_data[index].append(auc)
            #       print(box_data)
            #plt.boxplot(box_data, patch_artist=True, tick_labels=models)
            plt.boxplot(box_data)
            plt.xticks(range(len(models)), models)
            plt.plot([1, len(models)], [0.5, 0.5], "r--")
            plt.xlabel("model")
            plt.ylabel(f"{score}")
            plt.title(category_labels[category][cat_idx])
            plt.ylim([0, 1])
            plt.savefig(f"outputs/box_{score}_{tv}_{category}_{cat_idx}_{consensus}.png")
            plt.close()

for score in ["auc", "acc", "ef1", "tnr", "tpr"]:
    for category in category_labels.keys():
        for cat_idx in range(len(category_labels[category])):
            models = list(data[tv][category].keys())
            if len(models) == 0:
                continue
            plt.clf()
            box_data = [[] for _ in range(len(models))]
            for index, name in enumerate(models):
                t_data = data["t"][category][name]["score"]
                v_data = data["v"][category][name]["score"]
                aucs_t = np.array(t_data[score])[:, cat_idx]
                aucs_v = np.array(v_data[score])[:, cat_idx]
                plt.scatter(aucs_t, aucs_v, label=name)
            plt.legend()
            # plt.plot([0,1],[0,1], 'r--')
            plt.xlabel(f"{score} training")
            plt.ylabel(f"{score} validation")
            plt.title(category_labels[category][cat_idx])
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.savefig(f"outputs/cov_{score}_{category}_{cat_idx}_{consensus}.png")
            plt.close()

for tv, category in product(usages, list(category_labels.keys())):
    models = list(data[tv][category].keys())
    if len(models) == 0:
        continue
    box_data = [[] for _ in range(len(models))]
    for index, name in enumerate(models):
        _data = data["v"][category][name]["total_accuracy"]
        aucs = np.array(_data)[:]
        amax = np.argmax(aucs)
        _data = data[tv][category][name]["total_accuracy"]
        aucs = np.array(_data)[:]
        box_data[index].append((name, float(aucs.flatten()[amax])))
    print(tv, category, consensus)
    print(np.array(box_data))

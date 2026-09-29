from pickle import load
from matplotlib import pyplot as plt
import numpy as np
from itertools import combinations
from itertools import product

category_labels = {
    "tn": ["Non-Triple Negative", "Triple Negative"],
    "stage_basic": ["I", "II", "III"],
    "grade": ["1", "2", "3"],
}


def _round(v: str) -> str:
    rnd = int(float(v) * 1000)
    rnd /= 10
    rnd = int(round(rnd))
    return f"{rnd}"


data = {}
tdata = {}

# The results are spread across two scores file
## I was suspicious of the scores.pickle point cloud results being so high
##  After a bit of digging, with the introduction of the `grade` category, a few images
##  were moved from the training to validation dataset, which would likely boost some
##  scores. The new scores.pickle has the results from the retrained models on the
##  updated dataset usages, making those the actual scores for those.
##  This change decreased the performance of some models and boosted some others.
## The energetics results did not change appreciablly
## Due to time constraints and lack of improved performance, image models will be
##  update in the future.
## The raw results files indicate no appreciable change expected

with open("outputs/scores.pickle.orig", "rb") as fhandle:
    data = load(fhandle)

_models = [
    ["image", "image_vt"],
    ["energetics"],
]
parameters = product(
    ["total_accuracy"],
    range(len(_models)),
    ["tn", "stage_basic", "grade"],
    [1, 9],
    ["v"],
)

# for consensus, tv, category, models_index, var in parameters:
for var, models_index, category, consensus, tv in parameters:
    models = _models[models_index]
    if len(models) == 0:
        continue
    plt.clf()
    # print(models_index, models)
    box_data = [[] for _ in range(len(models))]
    max_data = []
    for index, name in enumerate(models):
        # print(index, name)
        _data = data[consensus]["v"][category][name]["total_accuracy"]
        aucs = np.array(_data)[:]
        amax = np.argmax(aucs)
        _data = data[consensus][tv][category][name]["total_accuracy"]
        aucs = np.array(_data)[:]
        max_data.append((name, consensus, category, aucs[amax]))
        for auc in aucs.flatten():
            box_data[index].append(auc)
    #       print(box_data)
    plt.violinplot(box_data)
    plt.xticks(range(1, len(models) + 1), models, rotation=45)
    plt.plot(
        [1 - 0.5, len(models) + 0.5],
        [1 / len(category_labels[category]), 1 / len(category_labels[category])],
        "r--",
    )
    plt.ylim([0, 1])
    plt.xlabel("Model")
    plt.ylabel(f"Total Accuracy")
    plt.title(category)
    plt.savefig(f"outputs/{var}_{category}_{models_index}_{tv}_{consensus}.png")
    plt.xticks(range(1, len(models) + 1), ["" for _ in range(len(models))])
    _TICKS = np.arange(0, 1, 0.1)
    plt.yticks(_TICKS, ["" for _ in range(len(_TICKS))])
    plt.title("")
    plt.xlabel("")
    plt.ylabel("")
    plt.savefig(f"outputs/{var}_{category}_{models_index}_{tv}_{consensus}.svg")
    plt.close()

    max_data = np.array(max_data)
    for row in max_data:
        _model, _cons, _type, _val = row.flatten()
        _model = str(_model)
        _cons = int(_cons)
        if _model not in tdata.keys():
            tdata[_model] = []
        tdata[_model].append(f"{_round(_val)}\\%")
with open("outputs/scores.pickle", "rb") as fhandle:
    data = load(fhandle)
# Total accuracies

_models = [
    # ["image", "image_vt"],
    # ["energetics"],
    ["metrics", "nuclei_only", "nuclei", "metricenergetics", "latent"],
]
parameters = product(
    ["total_accuracy"],
    range(len(_models)),
    # ["tn", "stage_basic", "grade"],
    ["tn", "stage_basic"],
    [1, 9],
    ["v"],
)

# for consensus, tv, category, models_index, var in parameters:
for var, models_index, category, consensus, tv in parameters:
    models = _models[models_index]
    if len(models) == 0:
        continue
    plt.clf()
    # print(models_index, models)
    box_data = [[] for _ in range(len(models))]
    max_data = []
    for index, name in enumerate(models):
        # print(index, name)
        _data = data[consensus]["v"][category][name]["total_accuracy"]
        aucs = np.array(_data)[:]
        amax = np.argmax(aucs)
        _data = data[consensus][tv][category][name]["total_accuracy"]
        aucs = np.array(_data)[:]
        max_data.append((name, consensus, category, aucs[amax]))
        for auc in aucs.flatten():
            box_data[index].append(auc)
    #       print(box_data)
    plt.violinplot(box_data)
    plt.xticks(range(1, len(models) + 1), models, rotation=45)
    plt.plot(
        [1 - 0.5, len(models) + 0.5],
        [1 / len(category_labels[category]), 1 / len(category_labels[category])],
        "r--",
    )
    plt.ylim([0, 1])
    plt.xlabel("Model")
    plt.ylabel(f"Total Accuracy")
    plt.title(category)
    plt.savefig(f"outputs/{var}_{category}_{models_index+2}_{tv}_{consensus}.png")
    plt.xticks(range(1, len(models) + 1), ["" for _ in range(len(models))])
    _TICKS = np.arange(0, 1, 0.1)
    plt.yticks(_TICKS, ["" for _ in range(len(_TICKS))])
    plt.title("")
    plt.xlabel("")
    plt.ylabel("")
    plt.savefig(f"outputs/{var}_{category}_{models_index+2}_{tv}_{consensus}.svg")
    plt.close()

    max_data = np.array(max_data)
    for row in max_data:
        _model, _cons, _type, _val = row.flatten()
        _model = str(_model)
        _cons = int(_cons)
        if _model not in tdata.keys():
            tdata[_model] = []
        tdata[_model].append(f"{_round(_val)}\\%")

with open("outputs/scores.pickle.orig", "rb") as fhandle:
    data = load(fhandle)
parameters = product(
    ["total_accuracy"],
    range(len(_models)),
    # ["tn", "stage_basic", "grade"],
    ["grade"],
    [1, 9],
    ["v"],
)

# for consensus, tv, category, models_index, var in parameters:
for var, models_index, category, consensus, tv in parameters:
    models = _models[models_index]
    if len(models) == 0:
        continue
    plt.clf()
    # print(models_index, models)
    box_data = [[] for _ in range(len(models))]
    max_data = []
    for index, name in enumerate(models):
        # print(index, name)
        _data = data[consensus]["v"][category][name]["total_accuracy"]
        aucs = np.array(_data)[:]
        amax = np.argmax(aucs)
        _data = data[consensus][tv][category][name]["total_accuracy"]
        aucs = np.array(_data)[:]
        max_data.append((name, consensus, category, aucs[amax]))
        for auc in aucs.flatten():
            box_data[index].append(auc)
    #       print(box_data)
    plt.violinplot(box_data)
    plt.xticks(range(1, len(models) + 1), models, rotation=45)
    plt.plot(
        [1 - 0.5, len(models) + 0.5],
        [1 / len(category_labels[category]), 1 / len(category_labels[category])],
        "r--",
    )
    plt.ylim([0, 1])
    plt.xlabel("Model")
    plt.ylabel(f"Total Accuracy")
    plt.title(category)
    plt.savefig(f"outputs/{var}_{category}_{models_index+2}_{tv}_{consensus}.png")
    plt.xticks(range(1, len(models) + 1), ["" for _ in range(len(models))])
    _TICKS = np.arange(0, 1, 0.1)
    plt.yticks(_TICKS, ["" for _ in range(len(_TICKS))])
    plt.title("")
    plt.xlabel("")
    plt.ylabel("")
    plt.savefig(f"outputs/{var}_{category}_{models_index+2}_{tv}_{consensus}.svg")
    plt.close()

    max_data = np.array(max_data)
    for row in max_data:
        _model, _cons, _type, _val = row.flatten()
        _model = str(_model)
        _cons = int(_cons)
        if _model not in tdata.keys():
            tdata[_model] = []
        tdata[_model].append(f"{_round(_val)}\\%")

_key2name = {
    "image": "CNN",
    "image_vt": "ViT",
    "energetics": "Energy",
    "metrics": "Morphology",
    "nuclei_only": "Nuclei",
    "nuclei": "Morphology \\& Nuclei",
    "metricenergetics": "Morphology \\& Energy",
    "latent": "Latent Embedding",
}
for key in tdata.keys():
    print(f"{_key2name[key]:<20} & {' & '.join(tdata[key])} \\\\")
#
# dta = data.copy()
# for consensus in [1,9]:
#    data = dta[consensus]
#    for tv, category in product(["v"], list(category_labels.keys())):
#        models = list(data[tv][category].keys())
#        if len(models) == 0:
#            continue
#        box_data = [[] for _ in range(len(models))]
#        for index, name in enumerate(models):
#            _data = data["v"][category][name]["total_accuracy"]
#            aucs = np.array(_data)[:]
#            amax = np.argmax(aucs)
#            _data = data[tv][category][name]["total_accuracy"]
#            aucs = np.array(_data)[:]
#            box_data[index].append((name, float(aucs.flatten()[amax])))
#        print(tv, category, consensus)
#        print(np.array(box_data))

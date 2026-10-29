#!/usr/bin/env python3
from tqdm.auto import tqdm
import pickle
import numpy as np
from random import seed
from sklearn.metrics import roc_auc_score
from histclass.tools.env import get_env
from histclass.extensions.glob_ext import globs

_K = 5
_SEED = 42
# _SEED = None
print(_SEED)

N_MAX = 1  # Maximum number of experts to add to pool in MoE
N_BOOTSTRAP = 50
N_SAMPLES = 100
N_CLASSES = 2  # All are binary categories
BAYESIAN = True
TOP_K = _K
_KEY = "V"
DATA_KEYS = [
    "ACC",
    "ACC_CI_L",
    "ACC_CI_H",
    "AUROC",
    "AUROC_CI_L",
    "AUROC_CI_H",
    "MCC",
    "MCC_CI_L",
    "MCC_CI_H",
    "Score",
]


BAYESIAN_SCALE = True
BAYESIAN_BEFORE = True
WEIGHTS = False
NORMALIZED_WEIGHTS = False
BS_T = "Scaled" if BAYESIAN_SCALE else "Unscaled"
BS_B = "Outside" if BAYESIAN_BEFORE else "Inside"
BS_N = "NormalizedWeights" if NORMALIZED_WEIGHTS else ""
BS_W = "Weights" if WEIGHTS else ""
PREFIX = f"{BS_T}{BS_B}{BS_N}{BS_W}-"


def errorbar(x, yerr, w=1):
    plt.plot([x, x], [yerr[0], yerr[1]], color="black")
    plt.plot([x - w, x + w], [yerr[0], yerr[0]], color="black")
    plt.plot([x - w, x + w], [yerr[1], yerr[1]], color="black")


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def compute_acc(matrix):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        accuracy = np.sum(np.diag(matrix)) / np.sum(matrix)
        confidences = np.diag(matrix) / np.sum(matrix, axis=0)
    return accuracy, confidences


def compute_mcc(matrix):
    import warnings

    tp = matrix[0][0]
    tn = matrix[1][1]
    fp = matrix[0][1]
    fn = matrix[1][0]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            rval = (tp * tn - fp * fn) / (
                (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
            ) ** 0.5
            if np.isfinite(rval):
                return rval
        return 0.0
    except:
        return 0.0


def _bootstrap(x, fptr, *args, **kwargs):
    from numpy import array, isfinite
    from scipy.stats import bootstrap

    x = (array(x),)
    ci = bootstrap(x, fptr, *args, **kwargs)
    a = float(fptr(x))
    b = float(ci.confidence_interval.low)
    if not np.isfinite(b):
        b = a
    c = float(ci.confidence_interval.high)
    if not np.isfinite(c):
        c = a
    return (a, b, c)


def read_model_metadata(model_name):
    from json import load

    model_file = f"ihc/models/classification/{model_name}/metadata.json"
    model_metadata = dict()
    with open(model_file, "r") as fhandle:
        model_metadata = load(fhandle)
    return model_metadata


if _SEED is None:
    _SEED = np.random.randint(10000)
    print(_SEED)
seed(_SEED)
np.random.seed(_SEED)

# For the sake of re-running this analysis, draw from outputs of each model running on each image 1000 times
# Structured dict as:
#  "T" or "V"  for training or validation
#  0 or 1 for class label of the images
#  Image name like "TMA00.A01" for image name
#  Enclosed is a 1000 length list of outputs
BASE = get_env("FILE_BASE").catch("raw_probabilities/").unwrap()
file_list = globs(BASE + "*.pickle")

moe_model_pool = []
model_data = dict()
full_data = dict()
for model_fpath in file_list:
    model_name = model_fpath.split("/")[-1][: -len(".pickle")]
    model_metadata = read_model_metadata(model_name)
    model_data[model_name] = {
        # Accuracy of the model
        "acc": model_metadata["accuracy"],
        # TPR of each classification label
        "conf": model_metadata["confidences"],
        # Boolean to determine it is in the MoE pool
        "moe": model_metadata["moe_model"],
    }
    if model_metadata["moe_model"]:
        moe_model_pool.append(model_name)
    with open(model_fpath, "rb") as fhandle:
        full_data[model_name] = pickle.load(fhandle)

model_data_li = dict()
# Compute each model's Accuracy, AUROC, and MCC on the validation dataset
for model in tqdm(full_data.keys()):
    category = model.split(".")[0].split("2")[-1]
    model_type = model.split("-")[0]
    raw_boot_data = []
    for _ in range(N_BOOTSTRAP):
        # Non-consensus
        matrix = np.array(
            [
                [
                    0,
                ]
                * N_CLASSES
            ]
            * N_CLASSES
        )
        data = full_data[model]["V"]
        preds = []
        for class_i in range(N_CLASSES):
            cdata = data[class_i]
            data_names = cdata.keys()
            data_indicies = {k: v for (k, v) in enumerate(data_names)}
            for _ in range(N_SAMPLES):
                p = [
                    0,
                ] * N_CLASSES
                dindex = np.random.randint(len(data_names))
                dname = data_indicies[dindex]
                ddata = cdata[dname]
                for _ in range(1):
                    index = np.random.randint(len(ddata))
                    mx = ddata[index]
                    p = [float(p[i] + mx[i]) for i in range(N_CLASSES)]
                t = class_i
                prob = softmax(p)
                p = np.argmax(p)
                preds.append((class_i, prob))
                matrix[t][p] += 1
        true, preds = zip(*preds)
        preds = np.array(preds)
        accuracy, confidences = compute_acc(matrix)
        nc_acc = float(accuracy)
        nc_mcc = float(compute_mcc(matrix))
        nc_auroc = float(roc_auc_score(true, preds[:, 1]))
        nc_score = (
            1.0
            - (
                (1 - nc_acc) ** 2.0
                + (1 - ((nc_mcc + 1.0) / 2.0)) ** 2.0
                + (1 - nc_auroc) ** 2.0
            )
            ** 0.5
        )
        # Consensus
        matrix = np.array(
            [
                [
                    0,
                ]
                * N_CLASSES
            ]
            * N_CLASSES
        )
        data = full_data[model]["V"]
        preds = []
        for class_i in range(N_CLASSES):
            cdata = data[class_i]
            data_names = cdata.keys()
            data_indicies = {k: v for (k, v) in enumerate(data_names)}
            for _ in range(N_SAMPLES):
                p = [
                    0,
                ] * N_CLASSES
                dindex = np.random.randint(len(data_names))
                dname = data_indicies[dindex]
                ddata = cdata[dname]
                for _ in range(9):
                    index = np.random.randint(len(ddata))
                    mx = ddata[index]
                    p = [float(p[i] + mx[i]) for i in range(N_CLASSES)]
                t = class_i
                prob = softmax(p)
                p = np.argmax(p)
                preds.append((class_i, prob))
                matrix[t][p] += 1
        true, preds = zip(*preds)
        preds = np.array(preds)
        accuracy, confidences = compute_acc(matrix)
        c_acc = float(accuracy)
        c_mcc = float(compute_mcc(matrix))
        c_auroc = float(roc_auc_score(true, preds[:, 1]))
        c_score = (
            1.0
            - (
                (1 - c_acc) ** 2.0
                + (1 - ((c_mcc + 1.0) / 2.0)) ** 2.0
                + (1 - c_auroc) ** 2.0
            )
            ** 0.5
        )
        raw_boot_data.append(
            (
                float(nc_acc),
                float(c_acc),
                float(nc_mcc),
                float(c_mcc),
                float(nc_auroc),
                float(c_auroc),
                float(nc_score),
                float(c_score),
            )
        )
    nc_accs, c_accs, nc_mccs, c_mccs, nc_aurocs, c_aurocs, nc_scores, c_scores = zip(
        *raw_boot_data
    )
    nc_acc_r = _bootstrap(nc_accs, np.mean, confidence_level=0.95)
    c_acc_r = _bootstrap(c_accs, np.mean, confidence_level=0.95)
    nc_auroc_r = _bootstrap(nc_aurocs, np.mean, confidence_level=0.95)
    c_auroc_r = _bootstrap(c_aurocs, np.mean, confidence_level=0.95)
    nc_mcc_r = _bootstrap(nc_mccs, np.mean, confidence_level=0.95)
    c_mcc_r = _bootstrap(c_mccs, np.mean, confidence_level=0.95)
    nc_score = np.mean(nc_scores)
    c_score = np.mean(c_scores)
    model_data.setdefault(category, {}).setdefault(model, {}).update(
        {
            "VBNC_ACC": nc_acc_r[0],
            "VBNC_ACC_CI_L": nc_acc_r[1],
            "VBNC_ACC_CI_H": nc_acc_r[2],
            "VBC_ACC": c_acc_r[0],
            "VBC_ACC_CI_L": c_acc_r[1],
            "VBC_ACC_CI_H": c_acc_r[2],
            "VBNC_MCC": nc_mcc_r[0],
            "VBNC_MCC_CI_L": nc_mcc_r[1],
            "VBNC_MCC_CI_H": nc_mcc_r[2],
            "VBC_MCC": c_mcc_r[0],
            "VBC_MCC_CI_L": c_mcc_r[1],
            "VBC_MCC_CI_H": c_mcc_r[2],
            "VBNC_ROC": nc_auroc_r[0],
            "VBNC_ROC_CI_L": nc_auroc_r[1],
            "VBNC_ROC_CI_H": nc_auroc_r[2],
            "VBC_ROC": c_auroc_r[0],
            "VBC_ROC_CI_L": c_auroc_r[1],
            "VBC_ROC_CI_H": c_auroc_r[2],
        }
    )
    model_data_li.setdefault(category, {}).setdefault(model_type, []).append(
        (
            *nc_acc_r,
            *nc_auroc_r,
            *nc_mcc_r,
            nc_score,
            *c_acc_r,
            *c_auroc_r,
            *c_mcc_r,
            c_score,
            model,
        )
    )

# Restructure
# From: Model -> T/V -> Class label -> Data name -> Array of probs
# To: Pred_type -> T/V -> Class label -> Data name -> Model -> Array of probs
restructured = {}
for a, b_dict in full_data.items():
    for b, c_dict in b_dict.items():
        for c, d_dict in c_dict.items():
            for d, e in d_dict.items():
                category = a.split(".")[0].split("2")[-1]
                restructured.setdefault(category, {}).setdefault(b, {}).setdefault(
                    c, {}
                ).setdefault(d, {})[a] = e
full_data = restructured

# Compute the Mixture of Experts Statistics with the MoE model pool
for class_type in tqdm(full_data.keys()):
    model_pool = model_data[class_type].keys()
    A = len(model_pool)
    model_pool = [m for m in model_pool if m in moe_model_pool]
    B = len(model_pool)
    model_info = [
        (
            mname,
            model_data[mname]["acc"],
        )
        for mname in model_pool
    ]
    model_info.sort(key=lambda x: x[1])
    model_info = model_info[-TOP_K:]
    model_pool = [m[0] for m in model_info]
    C = len(model_pool)
    print(f"{C} of {B} of {A}")

    raw_boot_data = []
    for _ in range(N_BOOTSTRAP):
        # Non-consensus
        data = full_data[class_type][_KEY]
        moe_predictions = []
        for class_i in range(N_CLASSES):
            cdata = data[class_i]
            data_names = cdata.keys()
            data_indicies = {k: v for (k, v) in enumerate(data_names)}
            for _ in range(N_SAMPLES):
                p = np.array(
                    [
                        0.0,
                    ]
                    * N_CLASSES
                )
                dindex = np.random.randint(len(data_names))
                dname = data_indicies[dindex]
                ddata = cdata[dname]
                for model in ddata.keys():
                    if model not in model_pool:
                        continue
                    for _ in range(1):
                        xindex = np.random.randint(len(ddata[model]))
                        mx = softmax(np.array(ddata[model][xindex]))
                        scales = [1, 1]
                        macc = model_data[model]["acc"]
                        scales = model_data[model]["conf"]
                        scales = np.nan_to_num(np.array(scales))
                        if BAYESIAN_SCALE:
                            if BAYESIAN_BEFORE:
                                p += scales * np.log(mx)
                            else:
                                p += np.log(mx * scales)
                        else:
                            if WEIGHTS:
                                if NORMALIZED_WEIGHTS:
                                    full_scales = np.nan_to_num(
                                        np.array(
                                            [
                                                model_data[m]["conf"]
                                                for m in ddata.keys()
                                                if m in model_pool
                                            ]
                                        )
                                    )
                                    scales = np.exp(scales) / np.exp(full_scales).sum(0)
                                else:
                                    pass
                            else:
                                scales = np.array([1, 1])
                            p += scales * mx
                p = softmax(p)
                moe_predictions.append((class_i, int(np.argmax(p)), p))

        matrix = np.array(
            [
                [
                    0,
                ]
                * 2
            ]
            * 2
        )
        for t, p, r in moe_predictions:
            matrix[t][p] += 1

        l, ipred, pred = zip(*moe_predictions)
        pred = np.array(pred)

        nc_acc = np.sum(np.diag(matrix)) / np.sum(matrix)
        nc_mcc = compute_mcc(matrix)
        nc_auroc = roc_auc_score(l, pred[:, 1])
        # Consensus
        data = full_data[class_type][_KEY]
        moe_predictions = []
        for class_i in range(N_CLASSES):
            cdata = data[class_i]
            data_names = cdata.keys()
            data_indicies = {k: v for (k, v) in enumerate(data_names)}
            for _ in range(N_SAMPLES):
                p = np.array(
                    [
                        0.0,
                    ]
                    * N_CLASSES
                )
                dindex = np.random.randint(len(data_names))
                dname = data_indicies[dindex]
                ddata = cdata[dname]
                for model in ddata.keys():
                    if model not in model_pool:
                        continue
                    for _ in range(9):
                        xindex = np.random.randint(len(ddata[model]))
                        mx = softmax(np.array(ddata[model][xindex]))
                        scales = [1, 1]
                        macc = model_data[model]["acc"]
                        scales = model_data[model]["conf"]
                        scales = np.nan_to_num(np.array(scales))
                        if BAYESIAN_SCALE:
                            if BAYESIAN_BEFORE:
                                p += scales * np.log(mx)
                            else:
                                p += np.log(mx * scales)
                        else:
                            if WEIGHTS:
                                if NORMALIZED_WEIGHTS:
                                    full_scales = np.nan_to_num(
                                        np.array(
                                            [
                                                model_data[m]["conf"]
                                                for m in ddata.keys()
                                                if m in model_pool
                                            ]
                                        )
                                    )
                                    scales = np.exp(scales) / np.exp(full_scales).sum(0)
                                else:
                                    pass
                            else:
                                scales = np.array([1, 1])
                            p += scales * mx
                p = softmax(p)
                moe_predictions.append((class_i, int(np.argmax(p)), p))

        matrix = np.array(
            [
                [
                    0,
                ]
                * 2
            ]
            * 2
        )
        for t, p, r in moe_predictions:
            matrix[t][p] += 1

        l, ipred, pred = zip(*moe_predictions)
        pred = np.array(pred)

        c_acc = np.sum(np.diag(matrix)) / np.sum(matrix)
        c_mcc = compute_mcc(matrix)
        c_auroc = roc_auc_score(l, pred[:, 1])
        raw_boot_data.append(
            (
                float(nc_acc),
                float(c_acc),
                float(nc_mcc),
                float(c_mcc),
                float(nc_auroc),
                float(c_auroc),
                0,
                0,
            )
        )
    nc_accs, c_accs, nc_mccs, c_mccs, nc_aurocs, c_aurocs, nc_scores, c_scores = zip(
        *raw_boot_data
    )
    nc_acc_r = _bootstrap(nc_accs, np.mean, confidence_level=0.95)
    c_acc_r = _bootstrap(c_accs, np.mean, confidence_level=0.95)
    nc_auroc_r = _bootstrap(nc_aurocs, np.mean, confidence_level=0.95)
    c_auroc_r = _bootstrap(c_aurocs, np.mean, confidence_level=0.95)
    nc_mcc_r = _bootstrap(nc_mccs, np.mean, confidence_level=0.95)
    c_mcc_r = _bootstrap(c_mccs, np.mean, confidence_level=0.95)
    nc_score = np.mean(nc_scores)
    c_score = np.mean(c_scores)
    model_data.setdefault(class_type, {}).setdefault("moe", {}).update(
        {
            "VBNC_ACC": nc_acc_r[0],
            "VBNC_ACC_CI_L": nc_acc_r[1],
            "VBNC_ACC_CI_H": nc_acc_r[2],
            "VBC_ACC": c_acc_r[0],
            "VBC_ACC_CI_L": c_acc_r[1],
            "VBC_ACC_CI_H": c_acc_r[2],
            "VBNC_MCC": nc_mcc_r[0],
            "VBNC_MCC_CI_L": nc_mcc_r[1],
            "VBNC_MCC_CI_H": nc_mcc_r[2],
            "VBC_MCC": c_mcc_r[0],
            "VBC_MCC_CI_L": c_mcc_r[1],
            "VBC_MCC_CI_H": c_mcc_r[2],
            "VBNC_ROC": nc_auroc_r[0],
            "VBNC_ROC_CI_L": nc_auroc_r[1],
            "VBNC_ROC_CI_H": nc_auroc_r[2],
            "VBC_ROC": c_auroc_r[0],
            "VBC_ROC_CI_L": c_auroc_r[1],
            "VBC_ROC_CI_H": c_auroc_r[2],
        }
    )
    model_data_li.setdefault(class_type, {}).setdefault("moe", []).append(
        (
            *nc_acc_r,
            *nc_auroc_r,
            *nc_mcc_r,
            nc_score,
            *c_acc_r,
            *c_auroc_r,
            *c_mcc_r,
            c_score,
            "moe",
        )
    )


# Extra foundation model info
## Prov-gigapath
model_data_li["tn"].setdefault("Prov-GIGAPATH", []).append(
    (
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.66054,
        0.6586049999999999,
        0.6626000000000001,
        0.7345489700000001,
        0.7321902568149412,
        0.7367508527366433,
        0.33135795783675215,
        0.3274275150691037,
        0.335530716009428,
        0,
        "Prov-GIGAPATH",
    )
)
model_data_li["hilostage"].setdefault("Prov-GIGAPATH", []).append(
    (
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.5119649999999999,
        0.51018,
        0.51369,
        0.54112314,
        0.5387528748622923,
        0.5431516498445874,
        0.026448463510695568,
        0.022359444636940735,
        0.030278473069467786,
        0,
        "Prov-GIGAPATH",
    )
)
model_data_li["hilograde"].setdefault("Prov-GIGAPATH", []).append(
    (
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.6942649999999999,
        0.6923622759264503,
        0.6963250000000001,
        0.77412225,
        0.7721176165305961,
        0.7761180861427721,
        0.39323443034290745,
        0.3893304835409297,
        0.39737219151281894,
        0,
        "Prov-GIGAPATH",
    )
)
model_data_li["tn"].setdefault("PRISM", []).append(
    (
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.6639099999999999,
        0.6619450000000001,
        0.66577,
        0.7662391900000003,
        0.7639699419275895,
        0.7684779075035398,
        0.3444958204143422,
        0.34034480563389014,
        0.348205077579137,
        0,
        "PRISM",
    )
)
model_data_li["hilostage"].setdefault("PRISM", []).append(
    (
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.49156500000000003,
        0.48948508653779244,
        0.49365999999999993,
        0.5362214199999998,
        0.5332011015208917,
        0.5393229585258207,
        -0.018782709532329193,
        -0.02347660937794296,
        -0.014165114199756136,
        0,
        "PRISM",
    )
)
model_data_li["hilograde"].setdefault("PRISM", []).append(
    (
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.6747450000000002,
        0.672675,
        0.6767200000000001,
        0.7644383899999998,
        0.7622974618937394,
        0.7665504957415684,
        0.35972866493874883,
        0.3554146968161036,
        0.3638116054628091,
        0,
        "PRISM",
    )
)


### Create bars with this data
import numpy as np
from matplotlib import pyplot as plt

# plt.rcParams["axes.linewidth"] = 6
plt.rcParams["axes.linewidth"] = 0
plt.rcParams["hatch.color"] = "00000033"
plt.rcParams["hatch.linewidth"] = 4.5

W = 0.48
IW = 0.15
LW = 4

colors = ["red", "blue", "green", "pink", "orange", "purple", "cyan", "darkviolet"]
colors = [
    "moccasin",
    "lightskyblue",
    "palegreen",
    "pink",
    "lavender",
    "lightgreen",
    "violet",
    "peachpuff",
]

extras = [
    "TITAN",
    "PRISM",
    "Prov-GIGAPATH",
]
model_names = {
    "image": "CNN",
    "image_vt": "ViT",
    "morphology": "Cell Shape",
    "nuclei": "Nuclear Shape",
    "morphologynuclei": "Morphology",
    "energetics": "Physics",
    "morphologyenergetics": "Physics \\& Cell Shape",
    "morphologyenergeticsnuclei": "Physics \\& Morphology",
    "moe": "Mixture of Experts",
    "Prov-GIGAPATH": "Prov-GigaPath",
    "PRISM": "PRISM",
}
cat_orders = [
    (
        "all",
        [
            ("image", "violet"),
            ("image_vt", "darkviolet"),
            ("morphology", "lightsteelblue"),
            ("nuclei", "royalblue"),
            ("morphologynuclei", "cornflowerblue"),
            ("energetics", "palegreen"),
            ("morphologyenergetics", "forestgreen"),
            ("morphologyenergeticsnuclei", "yellowgreen"),
            # ("TITAN", "moccasin"),
            ("moe", "moccasin"),
            ("Prov-GIGAPATH", "lightcoral"),
            ("PRISM", "maroon"),
        ],
    ),
]
cat_order = [
    "image",
    "image_vt",
    "morphology",
    "nuclei",
    "morphologynuclei",
    "energetics",
    "morphologyenergetics",
    "morphologyenergeticsnuclei",
] + extras

for gvalue in model_data_li.keys():
    cat_data = model_data_li[gvalue]
    for metric in ["ACC", "AUROC", "MCC"]:
        midx = DATA_KEYS.index(metric)
        for gname, cat_order in cat_orders:
            plt.close()
            plt.clf()
            _ = plt.subplots(figsize=(1.76, 4.02))
            W = 0.4
            IW = 0.15
            LW = 6
            xlabels = []
            index = 0
            for ivalue, icolor in cat_order:
                xlabels.append(ivalue)
                if ivalue not in cat_data.keys():
                    plt.bar(
                        index + W / 2,
                        90,
                        width=2 * W,
                        color="white",
                        edgecolor="black",
                        hatch="\\",
                    )
                    if metric == "MCC":
                        plt.bar(
                            index + W / 2,
                            -90,
                            width=2 * W,
                            color="white",
                            edgecolor="black",
                            hatch="\\",
                        )
                    index += 1
                    continue
                idf = cat_data[ivalue]
                idf_nc = sorted(idf, key=lambda x: x[3])
                idf_c = sorted(idf, key=lambda x: x[7])
                y0 = idf_nc[-1][midx] * 100
                y0_l = idf_nc[-1][midx + 1] * 100
                y0_h = idf_nc[-1][midx + 2] * 100
                y1 = idf_c[-1][midx + len(DATA_KEYS)] * 100
                y1_l = idf_c[-1][midx + len(DATA_KEYS) + 1] * 100
                y1_h = idf_c[-1][midx + len(DATA_KEYS) + 2] * 100
                plt.bar(
                    index + 0.25,
                    y1,
                    width=2 * W,
                    color=icolor,
                    linewidth=0,
                )
                errorbar(index + 0.25, [y1_l, y1_h], w=2 * W / 8)
                index += 1
            if metric == "MCC":
                plt.ylim(-100, 100)
                plt.plot([-W, index + W], [0, 0], "r--")
            else:
                plt.ylim(0, 100)
                plt.plot([-W, index + W], [50, 50], "r--")
            plt.xlim(0 - W / 2, index - W / 2)
            plt.xticks([], [])
            lims = plt.ylim()
            ticks = [lims[0]]
            if metric == "MCC":
                ticks.append(0)
            ticks.append(lims[1])
            plt.yticks([], [])
            plt.tight_layout()
            plt.savefig(f"FigurePieces/bars/{PREFIX}{gname}-{metric}-{gvalue}.png")
            plt.savefig(f"FigurePieces/bars/{PREFIX}{gname}-{metric}-{gvalue}.svg")


# Textual Output
for metric in ["ACC", "AUROC", "MCC"]:
    midx = DATA_KEYS.index(metric)
    print(metric)
    for gname, cat_order in cat_orders:
        for ivalue, icolor in cat_order:
            if ivalue == "moe":
                print("\\hline")
            print("%-40s" % (model_names[ivalue],), end="")
            for gvalue in ["hilostage", "hilograde", "tn"]:
                cat_data = model_data_li[gvalue]
                if ivalue not in cat_data.keys():
                    continue
                idf = cat_data[ivalue]
                idf_nc = sorted(idf, key=lambda x: x[3])
                idf_c = sorted(idf, key=lambda x: x[7])
                y0 = idf_nc[-1][midx] * 100
                y0_l = idf_nc[-1][midx + 1] * 100
                y0_h = idf_nc[-1][midx + 2] * 100
                y1 = idf_c[-1][midx + len(DATA_KEYS)] * 100
                y1_l = idf_c[-1][midx + len(DATA_KEYS) + 1] * 100
                y1_h = idf_c[-1][midx + len(DATA_KEYS) + 2] * 100
                if ivalue in ["moe", "Prov-GIGAPATH", "PRISM"]:
                    print(
                        " & \\multicolumn{2}{|c|}{[%5.2f %5.2f]} " % (y1_l, y1_h),
                        end="",
                    )
                else:
                    print(
                        " & [%5.2f %5.2f] & [%5.2f %5.2f] " % (y0_l, y0_h, y1_l, y1_h),
                        end="",
                    )
            print(" \\\\")
            if ivalue == "moe":
                print("\\hline")

# Sub-figures
cat_orders = [
    (
        "image",
        [
            ("image", "violet"),
            ("image_vt", "darkviolet"),
        ],
    ),
    (
        "shape",
        [
            ("morphology", "lightsteelblue"),
            ("nuclei", "royalblue"),
            ("morphologynuclei", "cornflowerblue"),
        ],
    ),
    (
        "physics",
        [
            ("energetics", "palegreen"),
            ("morphologyenergetics", "forestgreen"),
            ("morphologyenergeticsnuclei", "yellowgreen"),
        ],
    ),
]

for gvalue in model_data_li.keys():
    cat_data = model_data_li[gvalue]
    for metric in ["ACC", "AUROC", "MCC"]:
        midx = DATA_KEYS.index(metric)
        for gname, cat_order in cat_orders:
            plt.close()
            plt.clf()
            if gname != "all":
                _ = plt.subplots(figsize=(1.76, 4.02))
                W = 0.4
                IW = 0.15
                LW = 6
            else:
                W = 0.45
                IW = 0.25
                LW = 4
            xlabels = []
            index = 0
            for ivalue, icolor in cat_order:
                xlabels.append(ivalue)
                if ivalue not in cat_data.keys():
                    if ivalue in extras:
                        plt.bar(
                            index + W / 2,
                            90,
                            width=2 * W,
                            color="white",
                            edgecolor="black",
                            hatch="\\",
                        )
                        if metric == "MCC":
                            plt.bar(
                                index + W / 2,
                                -90,
                                width=2 * W,
                                color="white",
                                edgecolor="black",
                                hatch="\\",
                            )
                    else:
                        plt.bar(index, 90, width=W, color="white", edgecolor="black")
                        plt.bar(
                            index + W,
                            90,
                            width=W,
                            color="white",
                            edgecolor="black",
                            hatch="//",
                        )
                        if metric == "MCC":
                            plt.bar(
                                index, -90, width=W, color="white", edgecolor="black"
                            )
                            plt.bar(
                                index + W,
                                -90,
                                width=W,
                                color="white",
                                edgecolor="black",
                                hatch="//",
                            )
                    index += 1
                    continue
                idf = cat_data[ivalue]
                idf_nc = sorted(idf, key=lambda x: x[3])
                idf_c = sorted(idf, key=lambda x: x[7])
                y0 = idf_nc[-1][midx] * 100
                y0_l = idf_nc[-1][midx + 1] * 100
                y0_h = idf_nc[-1][midx + 2] * 100
                y1 = idf_c[-1][midx + len(DATA_KEYS)] * 100
                y1_l = idf_c[-1][midx + len(DATA_KEYS) + 1] * 100
                y1_h = idf_c[-1][midx + len(DATA_KEYS) + 2] * 100
                plt.bar(index, y0, width=W, linewidth=LW, color=icolor, edgecolor=None)
                plt.bar(
                    index + 0.5,
                    y1,
                    width=W,
                    color=icolor,
                    linewidth=2,
                    edgecolor="black",
                )
                errorbar(index, [y0_l, y0_h], w=W / 8)
                errorbar(index + 0.5, [y1_l, y1_h], w=W / 8)
                index += 1
            if metric == "MCC":
                plt.ylim(-100, 100)
                plt.plot([-W, index + W], [0, 0], "r--")
            else:
                plt.ylim(0, 100)
                plt.plot([-W, index + W], [50, 50], "r--")
            plt.xlim(0 - W / 2, index - W / 2)
            plt.xticks([], [])
            lims = plt.ylim()
            ticks = [lims[0]]
            if metric == "MCC":
                ticks.append(0)
            ticks.append(lims[1])
            plt.yticks([], [])
            plt.tight_layout()
            plt.savefig(f"FigurePieces/bars/{PREFIX}{gname}-{metric}-{gvalue}.png")
            plt.savefig(f"FigurePieces/bars/{PREFIX}{gname}-{metric}-{gvalue}.svg")

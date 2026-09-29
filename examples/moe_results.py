from histclass.backends.pytorch.models.utils import download_model
from histclass.interfaces.moe import MoEModels, MoEData
from histclass.tools.moe.sampling import *
from numpy import array, zeros, diag
from numpy.random import dirichlet, rand
from typing import Any
from tqdm.auto import tqdm
from json import dumps
import pandas as pd

N_pre = 100  # 100
N_post = 200  # 2000
ONE_PER_TYPE = True
N_CONSENSUS = 9


class BootstrappedMoEModels(MoEModels):
    def __init__(self, raw, acc=None, name_mapping=None, n_classes=2):
        # Mapping is name -> index
        self.n_classes = n_classes
        self.model_confidences = raw
        self.model_accuracies = None
        if acc is not None:
            self.model_accuracies = acc
        if name_mapping is None:
            self.name_mapping = {i: i for i in range(self.model_confidences.shape[0])}
        else:
            self.name_mapping = name_mapping

    def get(self, model=None):
        model_idx = model
        if model is not None:
            model_idx = self.name_mapping[model]
        confidences = self.model_confidences.copy()
        if self.model_accuracies is not None:
            confidences = confidences.T
            confidences *= self.model_accuracies
            confidences = confidences.T
        if model is not None:
            return confidences[model_idx, ...]
        return confidences


class BootstrappedMoEData(MoEData):
    def __init__(self, data, names, n_classes=2):
        self.data = data
        self.names = names
        self.n_classes = n_classes

    def random_label(self) -> int:
        from numpy import unique
        from numpy.random import choice

        data = self.data.copy()
        label_data = unique(data["Label"])
        label = choice(label_data)
        return label

    def data_from_label(self, model=None, label=None):
        from numpy import unique
        from numpy.random import choice
        from numpy.random import permutation

        data = self.data.copy()
        if model is not None:
            data = data[data["FullName"] == model]
        if label is None:
            label = self.random_label()
        data = data[data["Label"] == label]
        if data.shape[0] == 0:
            return None

        dataset_names = unique(data["DataName"])
        dataset_name = choice(dataset_names)
        data = data[data["DataName"] == dataset_name]
        NAME_KEY = "Name" if ONE_PER_TYPE else "FullName"
        models = unique(data[NAME_KEY])
        return_data = []
        for model in models:
            mdata = data[data[NAME_KEY] == model]
            mdata = mdata[
                ["FullName"] + [f"Prediction-{idx}" for idx in range(self.n_classes)]
            ]
            # In case we have multiple ids mixed together for
            #  the model, select only one data to use
            models = unique(mdata["FullName"])
            _model = models[choice(len(models.shape))]
            mdata = mdata[mdata["FullName"] == _model].to_numpy()
            permutations = permutation(mdata.shape[0])[:N_CONSENSUS]
            for idx in permutations:
                from numpy import double

                _mdata = mdata[idx, -self.n_classes :].astype(double)
                _model = mdata[idx, 0]
                return_data.append((_model, _mdata.flatten()))
        return (return_data, label, dataset_name)

    def sample(self, model=None, label=None):
        from numpy import array

        output = self.data_from_label(model=model, label=label)
        if output is None:
            return None
        data, label, dataset_name = output
        probs = []
        models = []
        for model, prob in data:
            probs.append(prob)
            models.append(model)
        probs = array(probs)
        return (probs, models, label, dataset_name)


class BootstrappedMoEDataParser:
    @staticmethod
    def get(data_source: str, validation_only=True, scale_with_accuracy=True):
        from pandas import read_pickle
        from numpy import argmax, unique, zeros
        from numpy.random import choice

        raw_data = read_pickle(data_source)
        raw_data["Name"] = (
            raw_data["ModelName"]
            + "2"
            + raw_data["Category"]
            + "."
            + raw_data["ModelId"]
        )
        raw_data["FullName"] = (
            raw_data["ModelName"]
            + "2"
            + raw_data["Category"]
            + "."
            + raw_data["ModelId"]
        )
        if "Label" not in raw_data.columns:
            raw_data["Label"] = argmax(
                raw_data[["Label-0", "Label-1", "Label-2"]], axis=-1
            )
        if "Prediction" not in raw_data.columns:
            raw_data["Prediction"] = argmax(
                raw_data[["Prediction-0", "Prediction-1", "Prediction-2"]]
            )
        # Restrict to Validation
        if validation_only:
            raw_data = raw_data[raw_data["Usage"] == 2]

        bm = []
        for category in unique(raw_data["Category"]):
            if not (("hilo" in category) or (category == "tn")):
                continue
            # if category == "tn":
            #    continue
            data = raw_data.copy()
            cdata = data[data["Category"] == category]
            n_classes = 3 if category in ["stage", "grade"] else 2
            model_names = list(set(unique(cdata["FullName"])))
            # model_names = list(filter(lambda x: not x.startswith("image"), model_names))
            # model_names = list(filter(lambda x: not x.startswith("image_vt"), model_names))
            # if category != "tn":
            #     model_names = list(filter(lambda x: not x.startswith("image"), model_names))
            # model_names = list(filter(lambda x: not x.startswith("latent"), model_names))
            cdata = cdata[cdata["FullName"].isin(model_names)]  # pyright: ignore
            confidences = zeros((len(model_names), n_classes))
            accuracy = zeros((len(model_names),))
            model_mapping = {
                model_name: midx for midx, model_name in enumerate(model_names)
            }
            model_name_arr = [""] * len(model_names)
            for mname, idx in model_mapping.items():
                model_name_arr[idx] = mname
            model_name_arr = array(model_name_arr)

            for midx, model_name in enumerate(model_names):
                model = download_model(model_name, metadata_only=True)
                confidences[midx, ...] = model.confidences
                accuracy[midx] = model.accuracy
                del model.model
                del model

            min_accuracy = 0.75
            mask = accuracy > min_accuracy

            confidences = confidences[mask]
            accuracy = accuracy[mask]
            model_name_arr = model_name_arr[mask]
            model_mapping = {
                model_name: midx
                for midx, model_name in enumerate(model_name_arr.tolist())
            }
            cdata = cdata[cdata["FullName"].isin(model_name_arr.tolist())]  # pyright: ignore

            if scale_with_accuracy:
                bmmm = BootstrappedMoEModels(
                    confidences,
                    accuracy,
                    name_mapping=model_mapping,
                    n_classes=n_classes,
                )
            else:
                bmmm = BootstrappedMoEModels(
                    confidences, name_mapping=model_mapping, n_classes=n_classes
                )
            bmmd = BootstrappedMoEData(cdata.copy(), model_names, n_classes=n_classes)
            bm.append((category, bmmm, bmmd))
        return bm


data_source = "MoeData.pickle"
data_source = "MoeData.boot.pickle"
data_source = "MoeDatas/MoeData.boot.pickle"
#data_source = "MoeDatas/.MoeDataBootPickleBackup"
data_source = "MoeDatas/Bootstrapped260525.pickle"
bmd = BootstrappedMoEDataParser.get(
    data_source, validation_only=True, scale_with_accuracy=False
)

from itertools import product

for cat, smm, smd in bmd:
    n_models = smm.model_confidences.shape[0]
    n_classes = smm.n_classes
    model_data = zeros((n_models, n_classes, n_classes))
    rev_map = {v: k for k, v in smm.name_mapping.items()}
    for _, midx, label in product(range(N_pre), range(n_models), range(n_classes)):
        model_name = rev_map[midx]
        sample_out = smd.sample(model=model_name, label=label)
        if sample_out is None:
            continue
        p_i = sample_out[0][0].flatten().argmax().item()
        l_i = sample_out[2]
        model_data[midx, l_i, p_i] += 1
    m_results = []
    for midx in range(n_models):
        m_results.append(
            diag(model_data[midx, ...]).sum() / model_data[midx, ...].sum()
        )
    m_results = array(m_results)
    top_k = m_results.argsort()[-5:]
    print(top_k)
    print([smd.names[idx] for idx in top_k])
    print(m_results[top_k])

# for f in [bayes_3, bayes_4, bayes_5]:
# for f in [bayes_1, bayes_2, bayes_3, bayes_4]:
for f in [bayes_3]:
    for cat, _smm, _smd in bmd:
        model_usage = {}
        data_usage = {}
        n_class = _smm.n_classes
        label_accuracies = zeros((_smm.n_classes, _smm.n_classes))
        for label in range(n_class):
            model_usage[label] = {}
            data_usage[label] = {}
            for _ in range(N_post):
                sample = smoe_run(_smm, _smd, f, label=label)
                if sample is None:
                    continue
                p_i, l_i, m, d, meta = sample
                used_model = meta["top_model"][meta["prediction"]]
                if used_model not in model_usage[label].keys():
                    model_usage[label][used_model] = {
                        "correct": 0,
                        "incorrect": 0,
                        "total": 0,
                    }
                is_correct = (
                    "correct" if meta["prediction"] == meta["label"] else "incorrect"
                )
                model_usage[label][used_model][is_correct] += 1
                model_usage[label][used_model]["total"] += 1
                if d not in data_usage[label].keys():
                    data_usage[label][d] = {"correct": 0, "incorrect": 0, "total": 0}
                data_usage[label][d][is_correct] += 1
                data_usage[label][d]["total"] += 1
                # print(meta)
                label_accuracies[l_i, p_i] += 1
        print(cat, f)
        columns = [
            f"Label-{label}-{kind}"
            for label, kind in product(range(_smm.n_classes), ["Accuracy", "Total"])
        ]
        model_results = pd.DataFrame(columns=columns)
        data_results = pd.DataFrame(columns=columns)
        # Get model usages
        for label in model_usage.keys():
            key = f"Label-{label}-"
            for model in model_usage[label].keys():
                model_results.loc[model, key + "Accuracy"] = (
                    int(
                        model_usage[label][model]["correct"]
                        / model_usage[label][model]["total"]
                        * 100_00
                    )
                    / 100.0
                )
                model_results.loc[model, key + "Total"] = model_usage[label][model][
                    "total"
                ]
        # Get data usages
        for label in data_usage.keys():
            key = f"Label-{label}-"
            for ds in data_usage[label].keys():
                data_results.loc[ds, key + "Accuracy"] = (
                    int(
                        data_usage[label][ds]["correct"]
                        / data_usage[label][ds]["total"]
                        * 100_00
                    )
                    / 100.0
                )
                data_results.loc[ds, key + "Total"] = data_usage[label][ds]["total"]
        print(model_results.to_string())
        print(data_results.to_string())
        print(cat, f)
        print(label_accuracies)
        print(diag(label_accuracies) / label_accuracies.sum(-1))
        print(diag(label_accuracies).sum() / label_accuracies.sum())

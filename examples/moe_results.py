from histclass.backends.pytorch.models.utils import download_model
from histclass.interfaces.moe import MoEModels, MoEData
from histclass.tools.moe.sampling import *
from numpy import array, zeros, diag
from numpy.random import dirichlet, rand
from typing import Any
from tqdm.auto import tqdm


class BootstrappedMoEModels(MoEModels):
    def __init__(self, raw, acc=None, n_classes=2):
        self.n_classes = n_classes
        self.model_confidences = raw
        self.model_accuracies = None
        if acc is not None:
            self.model_accuracies = acc

    def get(self, model=None):
        confidences = self.model_confidences.copy()
        if self.model_accuracies is not None:
            confidences = confidences.T
            confidences /= self.model_accuracies
            confidences = confidences.T
        if model is not None:
            return confidences[model, ...]
        return confidences


class BootstrappedMoEData(MoEData):
    def __init__(self, data, names, n_classes=2):
        self.data = data
        self.names = names
        self.n_classes = n_classes

    def sample(self, model=None, label=None, one=True):
        from numpy import array
        from numpy.random import choice

        data = self.data.copy()
        if model is not None:
            model_name = self.names[model]
            data = data[data["FullName"] == model_name]
        if label is not None:
            from numpy import unique

            data = data[data["Label"] == label]
            if data.shape[0] == 0:
                return
        if one:
            selected = choice(data.shape[0])
            data = data.iloc[[selected], ...]
        probs = data[[f"Prediction-{idx}" for idx in range(self.n_classes)]].to_numpy()
        model = data[["FullName"]].to_numpy()
        label = data[["Label"]].to_numpy()
        return (probs, model, label)


class BootstrappedMoEDataParser:
    @staticmethod
    def get(data_source: str, validation_only=True, scale_with_accuracy=True):
        from pandas import read_pickle
        from numpy import argmax, unique, zeros
        from numpy.random import choice

        raw_data = read_pickle(data_source)
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
            data = raw_data.copy()
            cdata = data[data["Category"] == category]
            n_classes = 2 if category == "tn" else 3
            model_names = list(set(unique(cdata["FullName"])))
            # model_names = list(filter(lambda x: not x.startswith("image"), model_names))
            # model_names = [model_names[0],model_names[1]]
            # model_names = [model_names[idx] for idx in choice(len(model_names), size=5)]
            confidences = zeros((len(model_names), n_classes))
            accuracy = zeros((len(model_names),))
            for midx, model_name in enumerate(model_names):
                model = download_model(model_name, metadata_only=True)
                confidences[midx, ...] = model.confidences
                accuracy[midx] = model.accuracy
                del model.model
                del model
            if scale_with_accuracy:
                bmmm = BootstrappedMoEModels(confidences, accuracy, n_classes=n_classes)
            else:
                bmmm = BootstrappedMoEModels(confidences, n_classes=n_classes)
            bmmd = BootstrappedMoEData(cdata.copy(), model_names, n_classes=n_classes)
            bm.append((category, bmmm, bmmd))
        return bm


data_source = "MoeData.pickle"
data_source = "MoeData.boot.pickle"
bmd = BootstrappedMoEDataParser.get(
    data_source, validation_only=True, scale_with_accuracy=True
)

# for f in [smoe_mix_1, smoe_gate_1, smoe_bayes_1, smoe_bayes_2]:
for f in [smoe_gate_1, smoe_bayes_1]:
    for cat, _smm, _smd in bmd:
        n_class = _smm.n_classes
        label_accuracies = zeros((_smm.n_classes, _smm.n_classes))
        for label in range(n_class):
            for _ in range(300):
                sample = smoe_run(_smm, _smd, f, label=label)
                if sample is None:
                    continue
                p_i, l_i = sample
                label_accuracies[l_i, p_i] += 1
        print(cat)
        print(f)
        print(label_accuracies)
        print(diag(label_accuracies) / label_accuracies.sum(-1))
        print(diag(label_accuracies).sum() / label_accuracies.sum())

from ...interfaces.moe import MoEModels, MoEData
from typing import Self, Any, Tuple, List, Dict
from numpy.typing import NDArray


class SyntheticMoEModels(MoEModels):
    def __init__(
        self: Self,
        raw: Any,
        acc: Any = None,
        name_mapping: Dict[str, Any] | None = None,
        n_classes: int = 2,
    ):
        self.model_confidences = raw
        self.n_classes = n_classes
        if acc is not None:
            self.model_accuracies = acc
        if name_mapping is None:
            self.name_mapping = {i: i for i in range(self.model_confidences.shape[0])}
        else:
            self.name_mapping = name_mapping

    def get_accuracy(self: Self, smd: MoEData):
        from numpy import zeros

        model_performances = zeros((self.model_confidences.shape[0], 2))
        for midx in range(self.model_confidences.shape[0]):
            for label in range(self.model_confidences.shape[1]):
                sample = smd.sample(model=midx, label=label)
                if sample is None:
                    continue
                prob, _m, l, _ = sample
                lp = prob.argmax(-1)
                tp = lp == l
                model_performances[midx, 0] += tp.sum()
                model_performances[midx, 1] += tp.shape[0]
        self.model_accuracies = model_performances[:, 0] / model_performances[:, 1]

    def get(self: Self, model: Any = None) -> NDArray[Any]:
        confidences = self.model_confidences.copy()
        if self.model_accuracies is not None:
            confidences = confidences.T
            confidences /= self.model_accuracies
            confidences = confidences.T
        if model is not None:
            return confidences[model, ...]
        return confidences


class SyntheticMoEData(MoEData):
    def sample(
        self: Self, model: Any = None, label: Any = None
    ) -> Tuple[NDArray[Any], List[Any], Any, Any]:
        from numpy import array

        data = self.data
        if model is not None:
            data = data[data[..., 0] == model]
        if label is not None:
            data = data[data[..., 1] == label]
        probs = data[..., -self.n_classes :]
        model = array(data[..., 0])
        label = array(data[..., 1])
        return (probs, data[..., 0].tolist(), label, "")

    @staticmethod
    def generate(smm: SyntheticMoEModels, n_samples: int = 100):
        from numpy import array

        # Task 1: Generate Synthetic Data
        #   Such that it follows with the confidences
        def simulate_prediction(conf: NDArray[Any], true_label: int):
            from numpy import asarray, ones
            from numpy.random import rand, choice, dirichlet

            conf = asarray(conf)  # pyright: ignore
            n_classes = len(conf)  # pyright: ignore

            # decide whether prediction is correct
            is_correct = rand() < conf[true_label]

            if is_correct:
                pred = true_label
            else:
                other = [i for i in range(n_classes) if i != true_label]
                pred = choice(other)

            # construct probability vector whose argmax is pred
            probs = dirichlet(ones(n_classes))
            probs[pred] += 1.0  # ensure argmax
            probs /= probs.sum()

            return probs

        model_predictions = []
        model_confidences = smm.model_confidences
        n_classes = model_confidences.shape[1]
        for midx, model in enumerate(model_confidences):
            for label in range(n_classes):
                sim = array(
                    [simulate_prediction(model, label) for _ in range(n_samples)]
                )
                for sample in sim:
                    model_predictions.append(  # pyright: ignore
                        (midx, label, sample.argmax(), *sample)
                    )
        model_predictions = array(model_predictions)  # pyright: ignore
        return SyntheticMoEData(model_predictions, n_classes=n_classes)

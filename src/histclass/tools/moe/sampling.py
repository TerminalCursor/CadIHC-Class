from ...interfaces.moe import MoEModels, MoEData
from numpy.typing import NDArray
from typing import Callable, Any, Tuple


def softmax(x: Any, axis: int | None = None):
    from numpy import exp

    ex = exp(x)
    _sum = ex.sum(axis=axis)
    return ex / _sum


def smoe_sample(
    smm: MoEModels, smd: MoEData, label: int | None = None
) -> Tuple[Any, int] | None:
    from numpy import array
    from numpy.random import randint

    if label is None:
        label = randint(smm.n_classes)
    try:
        model_probs = [  # pyright: ignore
            smd.sample(model=midx, label=label)[0]  # pyright: ignore
            for midx in range(smm.model_confidences.shape[0])
        ]
    except:
        return
    return array(model_probs).squeeze(), label  # pyright: ignore


def smoe_run(
    smm: MoEModels, smd: MoEData, f: Callable[[Any, Any], Any], label: int | None = None
) -> Tuple[int, int] | None:
    sample = smoe_sample(smm, smd, label=label)
    if sample is None:
        return
    p, l = sample
    p, _ = f(smm, p)
    return p, l


def smoe_mix_1(smm: MoEModels, probs: NDArray[Any]):
    confidence = smm.get()
    r = probs * confidence
    return r.max(0).argmax(-1).item(), r


def smoe_gate_1(smm: MoEModels, probs: NDArray[Any]) -> Tuple[Any, NDArray[Any]]:
    from numpy import array
    from numpy.random import choice

    confidence = smm.get()
    predictions = []
    for _label in range(smm.n_classes):
        _predictions = probs[:, _label]
        _confidences = softmax(confidence[:, _label])
        _choice = choice(  # pyright: ignore
            smm.model_confidences.shape[0], p=_confidences
        )
        predictions.append(_predictions[_choice])  # pyright: ignore
    r = array(predictions)  # pyright: ignore
    return r.argmax(-1).item(), r  # pyright: ignore


def smoe_bayes_1(smm: MoEModels, probs: NDArray[Any]):
    from numpy import exp, log

    eps = 1e-12
    confidence = smm.get()
    raw_predictions = probs
    model_weights = softmax(confidence.mean(axis=1))
    log_probs = log(raw_predictions + eps)
    weighted_log = log_probs * model_weights[:, None]
    bayes_scores = weighted_log.sum(axis=0)
    bayes_probs = exp(bayes_scores)
    bayes_probs /= bayes_probs.sum()
    prediction = bayes_probs.argmax()
    return prediction, bayes_probs


def smoe_bayes_2(smm: MoEModels, probs: NDArray[Any]):
    from numpy import exp, log

    eps = 1e-12
    confidence = smm.get()
    raw_predictions = probs
    log_pred = log(raw_predictions + eps)
    log_post = (confidence * log_pred).sum(axis=0)
    post = exp(log_post - log_post.max())
    post = post / post.sum()

    prediction = post.argmax().item()
    return prediction, post

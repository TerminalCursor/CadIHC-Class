from ...interfaces.moe import MoEModels, MoEData
from numpy.typing import NDArray
from typing import Callable, Any, Tuple, List, Dict


def softmax(x: Any, axis: int | None = None):
    from numpy import exp

    ex = exp(x)
    _sum = ex.sum(axis=axis)
    if axis == 1:
        return ex / _sum[:, None]
    return ex / _sum


def smoe_sample(
    smd: MoEData, label: int | None = None
) -> Tuple[NDArray[Any], List[Any], int, Any] | None:
    sample_outputs = smd.sample(label=label)
    return sample_outputs


def smoe_run(
    smm: MoEModels,
    smd: MoEData,
    f: Callable[[Any, Any], Tuple[int, NDArray[Any], Dict[str, Any]]],
    label: int | None = None,
) -> Tuple[int, int, List[Any], Any, Dict[str, Any]] | None:
    sample = smoe_sample(smd, label=label)
    if sample is None:
        return
    p, m, l, d = sample
    p, _, opt = f(smm, sample)
    return p, l, m, d, opt


def bayes_1(
    smm: MoEModels, sampled_data: Tuple[NDArray[Any], List[Any], NDArray[Any], Any]
) -> Tuple[int, NDArray[Any], Dict[str, Any]]:
    from numpy import exp, log, array

    probs = sampled_data[0]
    models = sampled_data[1]
    label = sampled_data[2]
    dataset = sampled_data[3]
    confidence = array([smm.get(model=model) for model in models])
    eps = 1e-12
    raw_predictions = probs
    model_weights = softmax(confidence.mean(axis=1))
    log_probs = log(raw_predictions + eps)
    weighted_log = log_probs * model_weights[:, None]
    bayes_scores = weighted_log.sum(axis=0)
    bayes_probs = exp(bayes_scores)
    bayes_probs /= bayes_probs.sum()
    prediction = bayes_probs.argmax()
    top_models = weighted_log.argmax(axis=0)
    top_models = [models[int(_model)] for _model in top_models]
    metadata = {
        "top_model": top_models,
        "raw_prediction": bayes_probs,
        "prediction": prediction.item(),
        "label": label,
        "dataset": dataset,
    }
    return prediction, bayes_probs, metadata


def bayes_2(
    smm: MoEModels, sampled_data: Tuple[NDArray[Any], List[Any], NDArray[Any], Any]
) -> Tuple[int, NDArray[Any], Dict[str, Any]]:
    from numpy import exp, log, array

    probs = sampled_data[0]
    models = sampled_data[1]
    label = sampled_data[2]
    dataset = sampled_data[3]
    confidence = array([smm.get(model=model) for model in models])
    eps = 1e-12
    raw_predictions = probs
    model_weights = softmax(confidence)
    log_probs = log(raw_predictions + eps)
    weighted_log = log_probs * model_weights
    bayes_scores = weighted_log.sum(axis=0)
    bayes_probs = exp(bayes_scores)
    bayes_probs /= bayes_probs.sum()
    prediction = bayes_probs.argmax()
    top_models = weighted_log.argmax(axis=0)
    top_models = [models[int(_model)] for _model in top_models]
    metadata = {
        "top_model": top_models,
        "raw_prediction": bayes_probs,
        "prediction": prediction.item(),
        "label": label,
        "dataset": dataset,
    }
    return prediction, bayes_probs, metadata


def bayes_3(
    smm: MoEModels, sampled_data: Tuple[NDArray[Any], List[Any], NDArray[Any], Any]
) -> Tuple[int, NDArray[Any], Dict[str, Any]]:
    from numpy import exp, log, array

    eps = 1e-12
    probs = sampled_data[0]
    models = sampled_data[1]
    label = sampled_data[2]
    dataset = sampled_data[3]
    confidence = array([smm.get(model=model) for model in models])
    raw_predictions = probs
    log_pred = log(raw_predictions + eps)
    weighted_log = confidence * log_pred
    log_post = weighted_log.sum(axis=0)
    post = exp(log_post - log_post.max())
    post = post / post.sum()

    prediction = post.argmax().item()
    top_models = weighted_log.argmax(axis=0)
    top_models = [models[int(_model)] for _model in top_models]
    metadata = {
        "top_model": top_models,
        "raw_prediction": post,
        "prediction": prediction,
        "label": label,
        "dataset": dataset,
    }
    return prediction, post, metadata


def moemix_1(
    smm: MoEModels, sampled_data: Tuple[NDArray[Any], List[Any], NDArray[Any], Any]
) -> Tuple[int, NDArray[Any], Dict[str, Any]]:
    from numpy import array

    probs = sampled_data[0]
    models = sampled_data[1]
    label = sampled_data[2]
    dataset = sampled_data[3]
    confidence = array([smm.get(model=model) for model in models])
    r = probs * confidence
    top_models = r.argmax(axis=0)
    top_models = [models[int(_model)] for _model in top_models]
    metadata = {
        "top_model": top_models,
        "raw_prediction": r.max(0),
        "prediction": r.max(0).argmax(-1).item(),
        "label": label,
        "dataset": dataset,
    }
    return r.max(0).argmax(-1).item(), r, metadata


def moegate_1(
    smm: MoEModels, sampled_data: Tuple[NDArray[Any], List[Any], NDArray[Any], Any]
) -> Tuple[int, NDArray[Any], Dict[str, Any]]:
    from numpy import array
    from numpy.random import choice

    probs = sampled_data[0]
    models = sampled_data[1]
    label = sampled_data[2]
    dataset = sampled_data[3]
    confidence = array([smm.get(model=model) for model in models])

    predictions = []
    top_models = []
    for _label in range(smm.n_classes):
        _predictions = probs[:, _label]
        _confidences = softmax(confidence[:, _label])
        _choice = choice(  # pyright: ignore
            smm.model_confidences.shape[0], p=_confidences
        )
        top_models.append(_choice)  # pyright: ignore
        predictions.append(_predictions[_choice])  # pyright: ignore
    r = array(predictions)  # pyright: ignore
    top_models = [models[int(_model)] for _model in top_models]  # pyright: ignore
    metadata = {  # pyright: ignore
        "top_model": top_models,
        "raw_prediction": r,
        "prediction": r.argmax(-1).item(),
        "label": label,
        "dataset": dataset,
    }
    return r.argmax(-1).item(), probs, metadata  # pyright: ignore

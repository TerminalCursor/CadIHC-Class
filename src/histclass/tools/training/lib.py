from typing import Any, List, Tuple
from numpy.typing import NDArray


def matrix2acc(matrix):
    from numpy import diag
    return diag(matrix).sum() / matrix.sum()


def matrix2mcc(matrix):
    assert matrix.shape[0] == 2
    assert matrix.shape[1] == 2
    TP = matrix[0,0]
    TN = matrix[1,1]
    FP = matrix[1,0]
    FN = matrix[0,1]
    return ((TP * TN) - (FP * FN)) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5

# Compute consensus on data: bootstrapped. Use the triple positive rate/ triple negative rate for confidence
def consensus(model, dataloaders, n_samples=1000, n_rounds=1, bootstrapped=False, raw_values=False):
    """# Get a consensus matrix"""
    from torch import zeros_like, float32, no_grad
    from numpy import array
    from tqdm.auto import tqdm
    from ...interfaces.monad.maybe import Maybe

    buckets = None
    raw_data = Maybe([]) if raw_values else Maybe()
    with no_grad():
        for _ in tqdm(range(n_samples)):
            for dataloader in dataloaders:
                # Bootstrapped means the dataloader only applies once
                for data, label, _ in dataloader:
                    if buckets == None:
                        n_class = label.shape[-1]
                        buckets = [[0 for _ in range(n_class)] for _ in range(n_class)]
                    data = data.to(model.device)
                    y_true = label.to(model.device)
                    y_trues = y_true.argmax(-1)
                    y_pred = zeros_like(y_true, dtype=float32)
                    # Smooth over n_rounds for consensus: Max Likelyhood
                    for _ in range(n_rounds):
                        y_pred += model.forward(data)
                    y_preds = y_pred.argmax(-1)
                    for y_true, y_pred_v, y_pred in zip(y_trues, y_preds, y_pred):
                        buckets[y_true][y_pred_v] += 1
                        if raw_data:
                            _v = y_true.item()
                            raw_data.data.append((y_pred.detach().cpu().numpy().tolist()[1], _v))
                        if bootstrapped:
                            break
                    if bootstrapped:
                        break
    if raw_data:
        return array(buckets), raw_data.data
    return array(buckets)


def to_arr(data: List[Tuple[Any]] | NDArray[Any]) -> NDArray[Any]:
    """
    Cast a datatype to a Numpy Array.

    If already a numpy array, ignore.

    Args:
        data (List[Tuple[Any]] | NDArray[Any]): The input data.

    Returns:
        NDArray[Any]: A numpy array representation of the input data.
    """
    from numpy import ndarray, array

    if isinstance(data, ndarray):
        return data
    return array(data)


def zero_slope(
    data: List[Tuple[Any]] | NDArray[Any],
    p_value: float = 0.05,
    alternative: str = "less",
) -> Tuple[bool, float]:
    """
    Determine if data has zero slope.

    Confidence level from `p_value`.

    Args:
        data (List[Tuple[Any]] | NDArray[Any]): The input data.
        p_value (float, optional): The confidence level. Defaults to 0.05.
        alternative (str, optional): The type of test. Defaults to "less".

    Returns:
        Tuple[bool, float]: A tuple containing the result and the slope value.
    """
    from scipy.stats import (  # pyright: ignore[reportMissingTypeStubs]
        linregress,  # pyright: ignore[reportUnknownVariableType]
    )

    data = to_arr(data)
    x = range(data.shape[-1])
    y = data
    reg_output = linregress(x, y, alternative=alternative)
    return reg_output.pvalue > p_value, reg_output.slope  # pyright: ignore


def transpose(data: List[Tuple[Any]] | NDArray[Any]) -> List[Any] | NDArray[Any]:
    """
    Transpose data.

    Converts between:
    [(a_1,b_1,c_1,...),(a_2,b_2,c_2,...),...]
    [(a_1,a_2,...),(b_1,b_2,...),...]

    Args:
        data (List[Tuple[Any]] | NDArray[Any]): The input data.

    Returns:
        List[Any] | NDArray[Any]: A transposed numpy array representation of the input data.
    """
    from numpy import ndarray

    if isinstance(data, ndarray):
        return data.T
    return list(zip(*data))


def mean_near(
    data: List[Tuple[Any]] | NDArray[Any], value: float | int, n_std: float = 1
) -> bool:
    """
    Determine if the mean is near a given value.

    Returns if the data is within `n_std` of the value.

    Args:
        data (List[Tuple[Any]] | NDArray[Any]): The input data.
        value (float | int): The target value.
        n_std (float, optional): The number of standard deviations. Defaults to 1.

    Returns:
        bool: A boolean indicating whether the mean is near the value.
    """
    from numpy import abs

    data = to_arr(data)
    data.flatten()
    m = data.mean()
    d = abs(m - value)
    s = data.std() * n_std
    return d < s


def early_exit_detector(
    losses: List[Tuple[Any]] | NDArray[Any],
    start_after: int = 100,
    lookback: int = 600,
    p_cutoff: float = 0.05,
    on_index: int = 0,
) -> Tuple[bool, float]:
    """
    Find when loss equilibriates.

    Find when the `on_index` loss has a slope of zero and is linear
    during a window of `lookback` after `start_after` iterations have occurred
    with a confidence of `p_cutoff`.

    Assume `losses` is of the form
    [(training_0,validation_0),(training_1,validation_1),...]

    Args:
        losses (List[Tuple[Any]] | NDArray[Any]): The input data.
        start_after (int, optional): The number of iterations to wait. Defaults to 100.
        lookback (int, optional): The number of previous samples to consider. Defaults to 600.
        p_cutoff (float, optional): The confidence level. Defaults to 0.05.
        on_index (int, optional): The index of the loss to check. Defaults to 0.

    Returns:
        Tuple[bool, float]: A tuple containing the result and the slope value.
    """
    data = transpose(losses)
    data = to_arr(data)
    data = data[on_index, ...]
    data = data[start_after:, ...]
    if len(data) < lookback:
        return False, 0
    data = data[-lookback:, ...]
    return zero_slope(data, p_value=p_cutoff)

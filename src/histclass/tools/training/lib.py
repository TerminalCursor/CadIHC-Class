from typing import Any, List, Tuple
from numpy.typing import NDArray


def to_arr(data: List[Tuple[Any]] | NDArray[Any]) -> NDArray[Any]:
    """# Cast a datatype to a Numpy Array

    If already a numpy array, ignore
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
    """# Determine if data has zero slope

    Confidence level from `p_value`
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
    """# Transpose data

    Converts between:
    [(a_1,b_1,c_1,...),(a_2,b_2,c_2,...),...]
    [(a_1,a_2,...),(b_1,b_2,...),...]
    """
    from numpy import ndarray

    if isinstance(data, ndarray):
        return data.T
    return list(zip(*data))


def mean_near(
    data: List[Tuple[Any]] | NDArray[Any], value: float | int, n_std: float = 1
) -> bool:
    """# Determine if the mean is near value

    Returns if the data is within `n_std` of the value
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
    """# Find when loss equilibriates

    Find when the `on_index` loss has a slope of zero and is linear
    during a window of `lookback` after `start_after` iterations have occurred
    with a confidence of `p_cutoff`

    Assume `losses` is of the form
    [(training_0,validation_0),(training_1,validation_1),...]
    """
    data = transpose(losses)
    data = to_arr(data)
    data = data[on_index, ...]
    data = data[start_after:, ...]
    if len(data) < lookback:
        return False, 0
    data = data[-lookback:, ...]
    return zero_slope(data, p_value=p_cutoff)

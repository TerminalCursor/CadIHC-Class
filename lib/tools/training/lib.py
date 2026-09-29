from ..call_graph import cg


def to_arr(data):
    """# Cast a datatype to a Numpy Array

    If already a numpy array, ignore
    """
    cg.add("to_arr")
    from numpy import ndarray, array

    if isinstance(data, ndarray):
        return data
    return array(data)


def zero_slope(data, p_value=0.05):
    """# Determine if data has zero slope

    Confidence level from `p_value`
    """
    cg.add("zero_slope")
    from scipy.stats import linregress

    data = to_arr(data)
    x = range(data.shape[-1])
    y = data
    reg_output = linregress(x, y)
    return reg_output.pvalue > p_value  # pyright: ignore


def transpose(data):
    """# Transpose data

    Converts between:
    [(a_1,b_1,c_1,...),(a_2,b_2,c_2,...),...]
    [(a_1,a_2,...),(b_1,b_2,...),...]
    """
    cg.add("transpose")
    from numpy import ndarray

    if isinstance(data, ndarray):
        return data.T
    return list(zip(*data))


def mean_near(data, value, n_std=1):
    """# Determine if the mean is near value

    Returns if the data is within `n_std` of the value
    """
    cg.add("mean_near")
    from numpy import abs

    data = to_arr(data)
    data.flatten()
    m = data.mean()
    d = abs(m - value)
    s = data.std() * n_std
    return d < s


def early_exit_detector(losses, t_i=100, window_len=600, p_cutoff=0.05, data_index=0):
    """# Find when loss equilibriates

    Find when the `data_index` loss has a slope of zero and is linear
    during a window of `window_len` after `t_i` iterations have occurred
    with a confidence of `p_cutoff`

    Assume `losses` is of the form
    [(training_0,validation_0),(training_1,validation_1),...]
    """
    cg.add("early_exit_detector")
    data = transpose(losses)
    data = to_arr(data)
    data = data[data_index, ...]
    data = data[t_i:, ...]
    if len(data) < window_len:
        return False
    return zero_slope(data, p_value=p_cutoff)

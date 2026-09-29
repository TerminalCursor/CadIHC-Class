from .metadata import Metadata
from torch import Tensor
from pandas import DataFrame  # pyright: ignore[reportMissingTypeStubs]
from numpy.typing import NDArray
from typing import List, Tuple, Any

# Normalization Attributes
normalization_attributes = {
    "metric": {
        "columns": [
            "Area",
            "MjrAxisLength",
            "MnrAxisLength",
            "Eccentricity",
            "ConvexArea",
            "EquivDiameter",
            "Solidity",
            "Extent",
            "Perimeter",
            "ConvexPerim",
            "FibLen",
            "InscribeR",
            "BlebLen",
        ],
        "means": [
            3502.971142073042,
            85.00468020299446,
            53.464603472173415,
            0.731229006178522,
            3845.033671099852,
            64.71664786018896,
            0.9183397951608545,
            0.6103669853421727,
            236.42698767152342,
            233.2487797762296,
            100.589953051465,
            23.846228548049524,
            34.359612821138484,
        ],
        "stdev": [
            1.85780382e03,
            2.46159747e01,
            1.55011250e01,
            1.46940730e-01,
            2.10144664e03,
            1.64886892e01,
            5.45856614e-02,
            9.06750294e-02,
            6.45156794e01,
            6.17974799e01,
            2.93619892e01,
            6.94590854e00,
            1.92704459e01,
        ],
        "max_scale": [
            6.18800000e03,
            1.52125653e02,
            8.19416135e01,
            9.99175882e-01,
            8.24700000e03,
            8.87626402e01,
            1.00000000e00,
            9.27272727e-01,
            5.78900000e02,
            3.58986000e02,
            1.72137177e02,
            3.04138126e01,
            1.54225449e02,
        ],
    },
    "nuclei": {
        "columns": [
            "area",
            "area_convex",
            "perimeter",
            "axis_major_length",
            "axis_minor_length",
            "eccentricity",
            "extent",
            "solidity",
            "equivalent_diameter_area",
        ],
        "means": [
            429.29368823387625,
            450.62477187424673,
            76.27442670437136,
            27.91251776669599,
            18.498285917806736,
            0.7022634386169229,
            0.7163965729747223,
            0.9515115374290878,
            22.34525824686273,
        ],
        "stdev": [
            3.20191002e02,
            3.39974166e02,
            2.51591279e01,
            9.05870226e00,
            6.33460309e00,
            1.54504718e-01,
            7.60154292e-02,
            2.61432521e-02,
            6.87627327e00,
        ],
    },
    "energetics": {
        "means": [
            3502.46273207,
            211.98549232,
            69.32343727,
            144.4087803,
        ],
        "stdev": [
            1856.88233949,
            58.47988665,
            59.3040725,
            67.68173487,
        ],
    },
}


def transform_image(
    im: NDArray[Any],
    baseline: List[Tuple[float, float]] = [
        (0.66652524, 0.1445229),
        (0.62870604, 0.18943928),
        (0.59433895, 0.22827524),
    ],
    im_range: Tuple[int, int] = (0, 1),
) -> NDArray[Any]:
    """# Transform Image to Baseline Mean and Stdev"""
    from numpy import clip, array

    chans: List[NDArray[Any]] = []
    transposed = False
    if im.shape[0] == 3:
        transposed = True
        im = im.T
    R, G, B = ((im - im_range[0]) / (im_range[1] - im_range[0])).reshape((-1, 3)).T
    for idx, channel in enumerate([R, G, B]):
        var_matched = channel * (baseline[idx][1] / channel.std())
        mean_matched = var_matched + (baseline[idx][0] - var_matched.mean())
        chans.append(clip(mean_matched, 0, 1))
    out_im = array(chans).T.reshape(im.shape)
    out_im = out_im * (im_range[1] - im_range[0]) + im_range[0]
    if transposed:
        out_im = out_im.T
    return out_im


def _rescale(
    x: NDArray[Any] | Tensor,
    m: List[Any] | NDArray[Any] | Tensor,
    s: List[Any] | NDArray[Any] | Tensor,
) -> NDArray[Any] | Tensor:
    """# Rescale data

    x -> (x - m) / s
    """
    from numpy import array

    m = array(m)
    s = array(s)
    return (x - m) / s


def normalize_nuclei(
    dataframe: DataFrame, metadata: Metadata, classic_norm: bool = False
) -> Tensor:
    """# Normalize Nuclei Metrics"""
    from numpy import array

    data_columns = array(normalization_attributes["nuclei"]["columns"])
    nuclei_means = array(normalization_attributes["nuclei"]["means"])
    nuclei_stdev = array(normalization_attributes["nuclei"]["stdev"])
    if classic_norm:
        return Tensor(dataframe[data_columns].to_numpy())  # pyright: ignore
    # New Scaling
    scaled_data = _rescale(
        dataframe[data_columns].to_numpy(),  # pyright: ignore
        nuclei_means,
        nuclei_stdev,
    )
    return Tensor(scaled_data)


def normalize_energetics(
    dataframe: NDArray[Any], metadata: Metadata, classic_norm: bool = False
) -> Tensor:
    """# Normalize Energetics"""
    from numpy import array

    energetics_means = array(normalization_attributes["energetics"]["means"])
    energetics_stdev = array(normalization_attributes["energetics"]["stdev"])
    if classic_norm:
        max_scale = array([26_000, 700, 550, 650])
        return Tensor(dataframe / max_scale)
    scaled_data = _rescale(
        dataframe, energetics_means, energetics_stdev  # pyright: ignore
    )
    return Tensor(scaled_data)


def normalize_metrics(
    dataframe: DataFrame, metadata: Metadata, classic_norm: bool = False
) -> Tensor:
    """# Normalize Morphology Metrics"""
    from numpy import array

    data_columns = array(normalization_attributes["metric"]["columns"])
    metric_means = array(normalization_attributes["metric"]["means"])
    metric_stdev = array(normalization_attributes["metric"]["stdev"])
    max_scale = array(normalization_attributes["metric"]["max_scale"])
    ratio = 10 / metadata.magnification
    prescale_powers = array([2, 1, 1, 0, 2, 1, 0, 0, 1, 1, 1, 1, 1])
    prescale = ratio**prescale_powers
    scaled_data = dataframe[data_columns].to_numpy() * prescale  # pyright: ignore
    max_scale = array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    scaled_data = scaled_data / max_scale  # pyright: ignore
    if classic_norm:
        return Tensor(scaled_data)
    # New Scaling
    scaled_data = _rescale(
        dataframe[data_columns].to_numpy(),  # pyright: ignore
        metric_means,
        metric_stdev,
    )
    return Tensor(scaled_data)


def normalize_metrics_nuclei(
    dataframe: DataFrame, metadata: Metadata, classic_norm: bool = False
) -> Tensor:
    """# Normalize Metrics and Nuclei

    Normalize both while retaining the correspondances
    """
    from torch import cat

    metrics_tensor = normalize_metrics(dataframe, metadata, classic_norm=classic_norm)
    nuclei_tensor = normalize_nuclei(dataframe, metadata, classic_norm=classic_norm)
    return cat([metrics_tensor, nuclei_tensor], dim=1)


def normalize_image(im: NDArray[Any], also_intensity: bool = False) -> Tensor:
    """# Normalize Image"""
    # Ensure W, H, and C channels
    assert len(im.shape) == 3
    # Ensure in the order CWH
    if im.shape[-1] <= 4:
        im = im.transpose(2, 0, 1)
    # Ignore Alpha Channel
    im = im[:3, ...]
    # Normalize Intensity
    if also_intensity:
        im = transform_image(im)
    return Tensor(im)

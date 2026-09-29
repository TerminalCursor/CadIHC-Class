from ....interfaces.data.types import DataTypes, CategoryType, DataframeType
from .normalization import *
from .metadata import Metadata
from torch import Tensor


def to_ohe(index: int, maximum: int) -> Tensor:
    """# Convert to One-hot encoding"""
    from torch import long
    from torch.nn.functional import one_hot

    return one_hot(Tensor([index]).to(long), maximum)


def get_data(
    source: str, dtype: DataTypes, metadata: Metadata, classic_norm: bool = False
) -> Tensor | None:
    """# Load the requested data from the source"""
    from os.path import exists

    if not dtype.is_valid():
        return None

    if dtype in [DataTypes.ImageCNN, DataTypes.ImageViT]:
        # Image Data For CNN or ViT
        #  Found in /image.png
        from matplotlib.pyplot import imread

        if not exists(source + "/image.png"):
            return None
        im = imread(source + "/image.png")
        im = normalize_image(im)
        return Tensor(im)

    elif dtype == DataTypes.Latent:
        # Latent Data
        #  Found in /MetricsAndLatent.csv
        from pandas import read_csv  # pyright: ignore[reportMissingTypeStubs]

        if not exists(source + "/MetricsAndLatent.csv"):
            return None
        data = read_csv(source + "/MetricsAndLatent.csv")
        if data.shape[0] == 0:
            return None
        return Tensor(data[["Latent-0", "Latent-1"]].to_numpy())  # pyright: ignore

    elif dtype == DataTypes.Morphology:
        # Morphology Data
        #  Found in /Metrics.csv
        from pandas import read_csv  # pyright: ignore[reportMissingTypeStubs]

        if not exists(source + "/Metrics.csv"):
            return None
        data = read_csv(source + "/Metrics.csv")
        if data.shape[0] == 0:
            return None
        return normalize_metrics(data, metadata, classic_norm=classic_norm)

    elif dtype == DataTypes.Energetics:
        # Energetics Data
        #  Found in /Energetics.npy
        from numpy import load as np_load

        if not exists(source + "/Energetics.npy"):
            return None
        data = np_load(source + "/Energetics.npy")
        if data.shape[0] == 0:
            return None
        return normalize_energetics(data, metadata, classic_norm=classic_norm)

    elif dtype == DataTypes.Morphology | DataTypes.Energetics:
        # Energetics Data
        #  Found in /Energetics.npy
        from numpy import load as np_load
        from pandas import read_csv  # pyright: ignore[reportMissingTypeStubs]
        from torch import cat

        if not exists(source + "/Metrics.csv"):
            return None
        if not exists(source + "/Energetics.npy"):
            return None
        data_m = read_csv(source + "/Metrics.csv")
        data_e = np_load(source + "/Energetics.npy")
        if data_m.shape[0] == 0:
            return None
        if data_e.shape[0] == 0:
            return None
        data_m = normalize_metrics(data_m, metadata, classic_norm=classic_norm)
        data_e = normalize_energetics(data_e, metadata, classic_norm=classic_norm)
        data_m = data_m[: min(data_e.shape[0], data_m.shape[0]), ...]
        data_e = data_e[: min(data_e.shape[0], data_m.shape[0]), ...]
        data = cat([data_m, data_e], dim=1)
        return data

    elif dtype == DataTypes.Nuclei:
        # Nuclei Data
        #  Found in /CNJoinedMetrics.csv
        from pandas import read_csv  # pyright: ignore[reportMissingTypeStubs]

        if not exists(source + "/CNJoinedMetrics.csv"):
            return None
        data = read_csv(source + "/CNJoinedMetrics.csv")
        if data.shape[0] == 0:
            return None
        return normalize_nuclei(data, metadata, classic_norm=classic_norm)

    elif dtype == DataTypes.Morphology | DataTypes.Nuclei:
        # Morphology and Nuclei Data
        #  Found in /CNJoinedMetrics.csv
        from pandas import read_csv  # pyright: ignore[reportMissingTypeStubs]

        if not exists(source + "/CNJoinedMetrics.csv"):
            return None
        data = read_csv(source + "/CNJoinedMetrics.csv")
        if data.shape[0] == 0:
            return None
        return normalize_metrics_nuclei(data, metadata, classic_norm=classic_norm)

    return None


def get_metadata_labels(
    source: str, ctype: CategoryType
) -> Tuple[Metadata, Tensor] | Tuple[None, None]:
    """# Get Labels as a One-Hot Encoding"""
    from os.path import exists
    from json import load

    if not exists(source + "/metadata.json"):
        return (None, None)
    metadata_dict = {}
    with open(source + "/metadata.json", "r") as fhandle:
        metadata_dict = load(fhandle)
    metadata = Metadata()
    if "magnification" in metadata_dict.keys():
        metadata.magnification = metadata_dict["magnification"]
    if "assignment" in metadata_dict.keys():
        if metadata_dict["assignment"] == "training":
            metadata.assignment = DataframeType.Training
        elif metadata_dict["assignment"] == "validation":
            metadata.assignment = DataframeType.Validation
    label_i = None
    if ctype == CategoryType.TripleNegative:
        label_i = metadata_dict.get("tn_i")
    elif ctype == CategoryType.BasicStage:
        label_i = metadata_dict.get("stage_basic_i")
    elif ctype == CategoryType.AdvancedStage:
        label_i = metadata_dict.get("stage_advanced_i")
    elif ctype == CategoryType.Grade:
        label_i = metadata_dict.get("grade_i")
    if label_i is None:
        return None, None
    label_vs = ctype.values()
    if label_vs is None:
        return None, None
    label_m = len(label_vs)
    if label_i < 0:
        return None, None
    labels = to_ohe(label_i, label_m).flatten()
    return metadata, labels

from ....interfaces.data.dataset import Dataframe
from ....interfaces.data.types import DataframeType, DataTypes, CategoryType
from torch import Tensor


class TorchDataframe(Dataframe):
    """
    A PyTorch data structure representing a dataset.

    Attributes:
        data_types (DataTypes): The type of data in the dataframe.
        category_type (CategoryType): The type of categorical values in the dataframe.
        source (str): The source of the dataset.
        name (str): The name of the dataset.
        data (Tensor): The tensor representing the input data.
        labels (Tensor): The tensor representing the target labels.
        usage (DataframeType): The type of dataset, e.g., train, val, test.
    """

    data_types: DataTypes
    category_type: CategoryType

    source: str
    name: str

    data: Tensor
    labels: Tensor
    usage: DataframeType = DataframeType.Unknown

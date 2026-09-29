from ....interfaces.data.dataset import Dataframe
from ....interfaces.data.types import DataframeType, DataTypes, CategoryType
from torch import Tensor


class TorchDataframe(Dataframe):
    """# Dataframe

    Contains:
    data: Tensor of [N, *Dd]
    label: Tensor of [N, Dl]
    usage: DataframeType to determine if it should be a train, val, or test set
    """

    data_types: DataTypes
    category_type: CategoryType

    source: str
    name: str

    data: Tensor
    labels: Tensor
    usage: DataframeType = DataframeType.Unknown

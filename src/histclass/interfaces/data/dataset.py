from ...tools.call_graph import cg
from ..config import CompleteConfig
from ..data.types import DataTypes, CategoryType, DataframeType
from typing import Self, List, Any


class Dataframe:
    """# Dataframe Interface

    data_types: DataTypes for Dataframe
    category_types: CategoryType for Dataframe

    source: Source of Dataframe
    name: Name of Dataframe

    data: Any - the datum
    label: Any - the labels
    usage: DataframeType to determine the use of dataframe
    """

    data_types: DataTypes
    category_type: CategoryType

    source: str
    name: str

    data: Any
    labels: Any
    usage: DataframeType = DataframeType.Unknown

    def __init__(
        self: Self,
        source: str,
        input_dtypes: DataTypes,
        output_dtype: CategoryType,
        name: str = "",
        data: Any = None,
        labels: Any = None,
        usage: DataframeType = DataframeType.Unknown,
    ):
        """# Get Data from Sources"""
        cg.add("Dataset.__init__")
        self.data_types = input_dtypes
        self.category_type = output_dtype
        self.source = source
        self.name = name
        self.data = data
        self.labels = labels
        self.usage = usage


class Dataset:
    """# Dataset Interface

    data_types: DataTypes for Dataset
    category_type: CategoryType for Dataset

    data_frames: List of Dataframes
    """

    data_types: DataTypes
    category_type: CategoryType

    data_frames: List[Dataframe]

    batch_size: int = 0

    def __init__(self: Self, sources: List[str], configuration: CompleteConfig):
        """# Get Data from Sources"""
        cg.add("Dataset.__init__")
        self.data_types = configuration.input_data_types
        self.category_type = configuration.output_data_type
        self.data_frames = []
        self.batch_size = configuration.batch_size

    def get_dataloaders(self: Self) -> List[Any]:
        """# Get Dataloaders"""
        raise NotImplementedError("Implement me!")

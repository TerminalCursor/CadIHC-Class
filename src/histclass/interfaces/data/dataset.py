from ..config import CompleteConfig
from ..data.types import DataTypes, CategoryType, DataframeType
from typing import Self, List, Any


class Dataframe:
    """
    # Dataframe Interface

    This class represents a data frame in the dataset. It contains the necessary information to
    determine how the data should be used.

    Attributes:
        data_types (DataTypes): The data types for this data frame.
        category_type (CategoryType): The category type for this data frame.
        source (str): The source of this data frame.
        name (str): The name of this data frame.
        data (Any): The datum.
        labels (Any): The labels.
        usage (DataframeType): The type of usage for this data frame.
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
        """
        Initializes a new instance of the Dataframe class.

        Args:
            source (str): The source of this data frame.
            input_dtypes (DataTypes): The data types for this data frame.
            output_dtype (CategoryType): The category type for this data frame.
            name (str, optional): The name of this data frame. Defaults to "".
            data (Any, optional): The datum. Defaults to None.
            labels (Any, optional): The labels. Defaults to None.
            usage (DataframeType, optional): The type of usage for this data frame. Defaults to DataframeType.Unknown.

        Raises:
            ValueError: If the input_dtypes and output_dtype do not match.
        """
        self.data_types = input_dtypes
        self.category_type = output_dtype
        self.source = source
        self.name = name
        self.data = data
        self.labels = labels
        self.usage = usage


class Dataset:
    """
    # Dataset Interface

    This class represents a dataset. It contains the necessary information to get the data.

    Attributes:
        data_types (DataTypes): The data types for this dataset.
        category_type (CategoryType): The category type for this dataset.
        data_frames (List[Dataframe]): A list of data frames in this dataset.
        batch_size (int): The batch size for this dataset.
    """

    data_types: DataTypes
    category_type: CategoryType

    data_frames: List[Dataframe]

    batch_size: int = 0

    def __init__(self: Self, sources: List[str], configuration: CompleteConfig):
        """
        Initializes a new instance of the Dataset class.

        Args:
            sources (List[str]): A list of sources for this dataset.
            configuration (CompleteConfig): The configuration for this dataset.

        Raises:
            ValueError: If the batch size is not set correctly.
        """
        self.data_types = configuration.input_data_types
        self.category_type = configuration.output_data_type
        self.data_frames = []
        self.batch_size = configuration.batch_size

    def get_dataloaders(self: Self) -> List[Any]:
        """
        Gets the dataloaders for this dataset.

        Returns:
            List[Any]: A list of dataloaders.

        Raises:
            NotImplementedError: This method is not implemented.
        """
        raise NotImplementedError("Implement me!")

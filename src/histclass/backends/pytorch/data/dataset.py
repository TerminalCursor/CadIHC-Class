from ....interfaces.data.dataset import Dataset as DatasetInterface
from ....interfaces.data.types import DataTypes, CategoryType, DataframeType
from ....interfaces.monad.maybe import Maybe
from ....interfaces.config import CompleteConfig
from ..models.interface import PyTorchModel
from ..types import DataT, DatasetT, CDatasetT, DataloaderT
from .utils import get_metadata_labels, get_data
from .dataframe import TorchDataframe
from typing import Self, List, Any
from torch.utils.data import DataLoader
from torch import Tensor


class PointcloudDataframe(DatasetT, TorchDataframe):
    """
    # Dataframe for Pointclouds

    Implements my interface and the Pytorch Dataset Interface.

    Parameters:
        min_seq_length (int): The smallest possible sequence length.
        seq_len (int): The size of the sequence.
        mask_p (float): Probability to mask some of the sequence.
    """

    min_seq_length: int = 10
    seq_len: int = 20
    mask_p: float = 0.2

    def to(self: Self, *args: Any) -> Self:
        """
        Move dataset to device and dtype.

        Args:
            *args (Any): Device and dtype to move the data to.
        Returns:
            Self: The dataframe with the data moved.
        """
        self.data = self.data.to(*args)
        self.labels = self.labels.to(*args)
        return self

    def __getitem__(self: Self, idx: int) -> DataT:
        """
        Get item from index.

        Args:
            idx (int): The index of the item.
        Returns:
            DataT: The item at the specified index.
        """
        from torch import zeros, randint, rand, randperm

        output_data = zeros(
            self.seq_len,
            *self.data.shape[1:],
            device=self.data.device,
            dtype=self.data.dtype,
        )
        r_len = randint(self.min_seq_length, self.seq_len, (1,)).item()
        n = self.seq_len
        if rand((1)).item() < self.mask_p:
            n = r_len
        n = min(self.seq_len, self.data.shape[0], n)
        output_data[:n, ...] = self.data[randperm(self.data.size()[0]), ...][:n, ...]
        return output_data, self.labels, self.name

    def __len__(self: Self) -> int:
        """
        Get the length of the dataframe.

        Returns:
            int: The length of the dataframe.
        """
        return 1


class ImageDataframe(DatasetT, TorchDataframe):
    """
    # Dataframe for Images

    Implements my interface and the Pytorch Dataset Interface.

    Parameters:
        height (int): The height of the image.
        width (int): The width of the image.
    """

    height: int = 2560
    width: int = 1920

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
        Initialize the dataframe.

        Args:
            source (str): The source of the image.
            input_dtypes (DataTypes): The types of the input data.
            output_dtype (CategoryType): The type of the output label.
            name (str): The name of the dataframe.
            data (Any): The data to be used for the dataframe.
            labels (Any): The labels to be used for the dataframe.
            usage (DataframeType): The usage of the dataframe.

        Returns:
            None
        """
        super(TorchDataframe, self).__init__(
            source, input_dtypes, output_dtype, name, data, labels, usage
        )
        self.width = self.data.shape[1]
        self.height = self.data.shape[2]

    def to(self: Self, *args: Any) -> Self:
        """
        Move dataset to device and dtype.

        Args:
            *args (Any): Device and dtype to move the data to.
        Returns:
            Self: The dataframe with the data moved.
        """
        self.data = self.data.to(*args)
        self.labels = self.labels.to(*args)
        return self

    def __getitem__(self: Self, idx: int) -> DataT:
        """
        Get item from index.

        Args:
            idx (int): The index of the item.
        Returns:
            DataT: The item at the specified index.
        """
        from torch.nn.functional import pad
        from torch import randint

        x_s = 0
        y_s = 0
        if self.data.shape[1] > self.width:
            x_s = randint(self.data.shape[1] - self.width, (1,)).item()
        if self.data.shape[2] > self.height:
            y_s = randint(self.data.shape[2] - self.height, (1,)).item()

        data = self.data
        data = data[:, x_s : x_s + self.width, y_s : y_s + self.height]
        data = pad(
            data,
            (
                0,
                self.height - data.shape[2],
                0,
                self.width - data.shape[1],
                0,
                0,
            ),
        )
        return data, self.labels, self.name

    def __len__(self: Self) -> int:
        """
        Get the length of the dataframe.

        Returns:
            int: The length of the dataframe.
        """
        return 1


class TorchDataset(DatasetInterface):
    """
    # Dataset

    Contains:
    data_types (DataTypes): The types of the data.
    category_type (CategoryType): The type of the labels.
    data_frames (List[DatasetT]): A list of dataframes loaded from sources.

    Parameters:
        sources (List[str]): A list of sources to load data from.
        configuration (CompleteConfig): The configuration for the dataset.
        classic_norm (bool): Whether to use classic normalization.
    """

    def __init__(
        self: Self,
        sources: List[str],
        configuration: CompleteConfig,
        classic_norm: bool = False,
    ):
        """
        Initialize the dataset.

        Args:
            sources (List[str]): A list of sources to load data from.
            configuration (CompleteConfig): The configuration for the dataset.
            classic_norm (bool): Whether to use classic normalization.

        Returns:
            None
        """
        super().__init__(sources, configuration)
        DataframeInit = (
            ImageDataframe
            if self.data_types in [DataTypes.ImageCNN, DataTypes.ImageViT]
            else PointcloudDataframe
        )
        for source in sources:
            metadata, labels = get_metadata_labels(source, self.category_type)
            if metadata is None:
                continue
            data = get_data(
                source,
                self.data_types,
                metadata,
                classic_norm=classic_norm,
            )
            if data is None:
                continue
            if labels is None:
                continue
            usage = metadata.assignment
            parts = source.split("/")
            name = ""
            if parts[-1] == "":
                if len(parts) > 1:
                    name = parts[-2]
            else:
                if len(parts) > 0:
                    name = parts[-1]
            data_frame = DataframeInit(
                source,
                self.data_types,
                self.category_type,
                name=name,
                data=data,
                labels=labels,
                usage=usage,
            )
            if self.data_types not in [DataTypes.ImageCNN, DataTypes.ImageViT]:
                assert isinstance(data_frame, PointcloudDataframe)
                assert configuration.pointcloud_config is not None
                data_frame.min_seq_length = configuration.pointcloud_config.min_points
                data_frame.seq_len = configuration.pointcloud_config.max_points
            else:
                assert isinstance(data_frame, ImageDataframe)
                assert configuration.image_values is not None
                data_frame.width = configuration.image_values.window_width
                data_frame.height = configuration.image_values.window_height
            self.data_frames.append(data_frame)

    def get_dataloaders(self: Self) -> List[DataloaderT | List[DataloaderT]]:
        """
        Get dataloaders.

        Returns:
            List[DataloaderT | List[DataloaderT]]: A list of dataloaders.
        """
        from torch import stack, bincount, double
        from torch.utils.data import WeightedRandomSampler, ConcatDataset

        dataloaders: List[DataloaderT | List[DataloaderT]] = []

        def get_dataloader(
            usage: DataframeType | None = None, label: int | None = None
        ) -> Maybe[DataloaderT]:
            """
            Get dataloader for usage.

            Args:
                usage (DataframeType | None): The usage of the data.
                label (int | None): The label of the data.

            Returns:
                Maybe[DataloaderT]: A maybe dataloader.
            """
            dataset = self.data_frames
            if usage is not None:
                dataset = [
                    self.data_frames[index]
                    for index in range(len(self.data_frames))
                    if self.data_frames[index].usage == usage
                ]
            if label is not None:
                dataset = [
                    dataset[index]
                    for index in range(len(dataset))
                    if dataset[index].labels[..., label].item() == 1
                ]
            if len(dataset) == 0:
                return Maybe()
            labels = stack([dataset[index].labels for index in range(len(dataset))])
            labels = labels.argmax(1)
            l_bincount = bincount(labels)
            l_weights_v = 1.0 / l_bincount
            l_weights = Tensor(
                [l_weights_v[labels[idx]] for idx in range(len(labels))]
            ).to(double)
            sampler = WeightedRandomSampler(
                l_weights, len(l_weights), replacement=True  # pyright: ignore
            )
            new_dataset: CDatasetT = ConcatDataset(dataset)  # pyright: ignore
            dataloader = DataLoader(
                new_dataset, batch_size=self.batch_size, sampler=sampler
            )
            return Maybe(dataloader)

        # Training Dataset
        dl = get_dataloader(DataframeType.Training)
        dataloaders.append(dl.unwrap())
        dl = get_dataloader(DataframeType.Validation)
        dataloaders.append(dl.unwrap())
        dl = get_dataloader()
        dataloaders.append(dl.unwrap())
        n_classes = self.category_type.values()
        assert n_classes is not None
        sub_dataloaders: List[DataloaderT] = []
        for label in range(len(n_classes)):
            dl = get_dataloader(DataframeType.Training, label)
            if dl.data is not None:
                sub_dataloaders.append(dl.data)
        dataloaders.append(sub_dataloaders)
        sub_dataloaders: List[DataloaderT] = []
        for label in range(len(n_classes)):
            dl = get_dataloader(DataframeType.Validation, label)
            if dl.data is not None:
                sub_dataloaders.append(dl.data)
        dataloaders.append(sub_dataloaders)
        return dataloaders


def load_data(
    configuration: CompleteConfig, local_base: str = "./", classic_norm: bool = False
):
    """
    Load data.

    Assumed schema:
    ihc/data/classification/
          dataset.id/
            metadata.json        -- Sample metadata
            image.png            -- Stained Sample Source Image
            Metrics.csv          -- Morphology Data
            MetricsAndLatent.csv -- Morphology Data
            CNJoinedMetrics.csv  -- Morphology and Nuclei Data
            Energetics.npy       -- Energetics Data

    Args:
        sources (List[str]): A list of sources to load data from.
        configuration (CompleteConfig): The configuration for the dataset.
        classic_norm (bool): Whether to use classic normalization.

    Returns:
        TorchDataset: The loaded dataset.
    """
    from ....extensions.glob_ext import globs

    base_directory = f"{local_base}/ihc/data/classification/"
    data_sources = globs(base_directory + "/*/")
    return TorchDataset(data_sources, configuration, classic_norm=classic_norm)


def load_data_from_model(model: PyTorchModel, local_base: str = "./") -> TorchDataset:
    """
    Load data associated to model.
    I.e. a nuclei model will automatically pull nuclei data.

    Args:
        model (PyTorchModel): The model to load data from.
        local_base (str): The base directory for the data.

    Returns:
        TorchDataset: The loaded dataset.
    """
    from ..models.vit import ViTModel
    from ....tools.configuration import load_config

    classic_norm = model.classic_data_norm
    data_type = model.data_types
    category_type = model.category_type
    model_type = model.model_type
    patches = 1
    if isinstance(model, ViTModel):
        patches = model.image_size
    config = load_config(data_type, category_type, model_type, patches=patches)
    return load_data(config, local_base=local_base, classic_norm=classic_norm)

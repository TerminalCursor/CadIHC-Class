from ....tools.call_graph import cg
from ....interfaces.config import CompleteConfig
from ....interfaces.data.dataset import Dataset
from ....interfaces.data.types import DataTypes
from ....interfaces.models.types import ModelType
from ....interfaces.models.model import Model
from .cnn import CnnModel
from .pointcloud import PointcloudModel
from .vit import ViTModel
from .gru import GruModel
from torch.nn.functional import elu
from typing import Callable


def load_model(
    dataset: Dataset, configuration: CompleteConfig, activation: Callable = elu
) -> Model:
    """# Load a model"""
    cg.add("load_model")
    assert configuration.input_data_types.is_valid()
    if configuration.input_data_types == DataTypes.ImageCNN:
        return CnnModel(dataset, configuration, activation)
    elif configuration.input_data_types == DataTypes.ImageViT:
        return ViTModel(dataset, configuration, activation)
    else:
        if configuration.model_type == ModelType.Attention:
            return PointcloudModel(dataset, configuration, activation)
        return GruModel(dataset, configuration, activation)

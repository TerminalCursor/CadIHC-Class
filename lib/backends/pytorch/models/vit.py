from ....tools.call_graph import cg
from ....interfaces.config import CompleteConfig
from ....interfaces.data.dataset import Dataset
from .interface import PyTorchModel
from ..data.dataset import ImageDataframe
from typing import Self, Callable, Any
from torch.nn import Module
from torch.nn.functional import elu
from torchvision.models.vision_transformer import VisionTransformer


class ViTModel(PyTorchModel):
    """# ViT Image Model"""

    model: Module
    optimizer: Callable
    loss_fn: Callable

    def __init__(
        self: Self,
        dataset: Dataset,
        configuration: CompleteConfig,
        activation: Callable = elu,
    ):
        """# Model Initialization"""
        cg.add("ViTModel.__init__")
        assert isinstance(dataset.data_frames[0], ImageDataframe)
        assert configuration.image_values is not None
        assert configuration.vision_transformer_config is not None
        n_classes = dataset.data_frames[0].labels.shape[-1]
        self.model = VisionTransformer(
            num_classes=n_classes, **configuration.vision_transformer_config.to_dict()
        )

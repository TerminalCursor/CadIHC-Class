from ....tools.call_graph import cg
from ....interfaces.config import CompleteConfig
from ....interfaces.data.types import DataTypes, CategoryType
from ....interfaces.models.types import ModelType
from ....interfaces.models.types import LossT, OptimizerT
from .interface import PyTorchModel
from ..types import ActivationT
from typing import Self
from torch.nn import Module
from torch.nn.functional import elu
from torchvision.models.vision_transformer import (  # pyright: ignore[reportMissingTypeStubs]
    VisionTransformer,
)


class ViTModel(PyTorchModel):
    """# ViT Image Model"""

    model: Module
    optimizer: OptimizerT
    loss_fn: LossT

    model_type: ModelType = ModelType.Attention
    data_types: DataTypes
    category_type: CategoryType

    def __init__(
        self: Self,
        configuration: CompleteConfig,
        activation: ActivationT = elu,
    ):
        """# Model Initialization"""
        cg.add("ViTModel.__init__")
        assert configuration.image_values is not None
        assert configuration.vision_transformer_config is not None
        classes = configuration.output_data_type.values()
        assert classes is not None
        n_classes = len(classes)
        self.data_types = configuration.input_data_types
        self.category_type = configuration.output_data_type
        self.model = VisionTransformer(
            num_classes=n_classes,
            **configuration.vision_transformer_config.to_dict(),  # pyright: ignore
        )

from ....interfaces.config import CompleteConfig
from ....interfaces.data.types import DataTypes, CategoryType
from ....interfaces.models.types import ModelType
from ....interfaces.models.types import LossT, OptimizerT
from .interface import PyTorchModel
from ..types import ActivationT
from typing import Self
from torch.nn import Module
from torch.nn.functional import elu


class MixtureOfExperts(PyTorchModel):
    """# Mixture of Experts Model"""

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
        self.data_types = configuration.input_data_types
        self.category_type = configuration.output_data_type

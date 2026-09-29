from ....tools.call_graph import cg
from ....interfaces.config import CompleteConfig
from ....interfaces.data.dataset import Dataset
from .interface import PyTorchModel
from ..data.dataset import ImageDataframe
from typing import Self, Callable
from torch.nn import Module, Sequential
from torch import Tensor
from torch.nn.functional import elu
from torch import nn


class SimpleConvBlock(Module):
    """# Simple Hidden Layer"""

    def __init__(self: Self, n: int, m: int, activation: Callable = elu):
        """# Simple Conv Init"""
        cg.add("SimpleConvBlock.__init__")
        super(SimpleConvBlock, self).__init__()
        self.conv = Sequential(
            nn.Conv2d(n, m, kernel_size=3, padding=1),
            nn.BatchNorm2d(m),
        )
        self.activation = activation
        self.pool = Sequential(nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self: Self, x: Tensor) -> Tensor:
        """# Simple Conv Forward"""
        cg.add("SimpleConvBlock.forward")
        x = self.conv(x)
        x = self.activation(x)
        x = self.pool(x)
        return x


class SimpleCNN(Module):
    def __init__(self: Self, num_classes: int, activation: Callable = elu):
        """# Simple CNN PyTorch Init"""
        cg.add("SimpleCNN.__init__")
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            SimpleConvBlock(3, 8, activation=activation),
            SimpleConvBlock(8, 16, activation=activation),
            SimpleConvBlock(16, 32, activation=activation),
            # Adaptive pooling to handle arbitrary input sizes
            nn.AdaptiveAvgPool2d((1, 1)),  # Output feature map size is 1x1
        )
        self.classifier = nn.Sequential(nn.Linear(32, num_classes))

    def forward(self: Self, x: Tensor) -> Tensor:
        """# Simple CNN Forward"""
        cg.add("SimpleCNN.forward")
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)
        return x


class CnnModel(PyTorchModel):
    """# CNN Image Model"""

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
        cg.add("CnnModel.__init__")
        assert isinstance(dataset.data_frames[0], ImageDataframe)
        assert configuration.image_values is not None
        assert configuration.vision_transformer_config is None
        n_classes = dataset.data_frames[0].labels.shape[-1]
        self.model = SimpleCNN(n_classes, activation=activation)

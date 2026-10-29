from ....interfaces.config import CompleteConfig
from ....interfaces.data.types import DataTypes, CategoryType
from ....interfaces.models.types import ModelType
from ....interfaces.models.types import LossT, OptimizerT
from .interface import PyTorchModel
from ..types import ActivationT, DataloaderT
from typing import Self, Any
from torch.nn import Module, Sequential
from torch import Tensor
from torch.nn.functional import elu
from torch import nn


class SimpleConvBlock(Module):
    """
    A simple hidden layer in the CNN model.

    This module consists of a convolutional block with a max pooling layer.
    The convolutional block uses a ReLU activation function by default, but this can be customized.
    """

    def __init__(self: Self, n: int, m: int, activation: ActivationT = elu):
        """
        Initializes the SimpleConvBlock.

        Args:
            n (int): The number of input channels in the convolutional layer.
            m (int): The number of output channels in the convolutional layer.
            activation (ActivationT): The activation function to use. Defaults to ReLU.

        Raises:
            ValueError: If n or m is negative.
        """
        super(SimpleConvBlock, self).__init__()
        self.conv = Sequential(
            nn.Conv2d(n, m, kernel_size=3, padding=1),
            nn.BatchNorm2d(m),
        )
        self.activation = activation
        self.pool = Sequential(nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self: Self, x: Tensor) -> Tensor:
        """
        The forward pass of the SimpleConvBlock.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after passing through the convolutional block and max pooling layer.
        """
        x = self.conv(x)
        x = self.activation(x)
        x = self.pool(x)
        return x


class SimpleCNN(Module):
    """
    A simple CNN model using a single layer of convolutional blocks.

    This model consists of four layers of convolutional blocks with adaptive pooling at the end.
    The output feature map size is 1x1, which can be used as input to other models or classifiers.
    """

    def __init__(self: Self, num_classes: int, activation: ActivationT = elu):
        """
        Initializes the SimpleCNN.

        Args:
            num_classes (int): The number of output classes.
            activation (ActivationT): The activation function to use. Defaults to ReLU.

        Raises:
            ValueError: If num_classes is negative.
        """
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
        """
        The forward pass of the SimpleCNN.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after passing through the convolutional blocks and classifier.
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)
        return x


class CnnModel(PyTorchModel):
    """
    A CNN image model that can be customized with different models and configurations.
    """

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
        """
        Initializes the CnnModel.

        Args:
            configuration (CompleteConfig): The configuration for this model.
            activation (ActivationT): The activation function to use. Defaults to ReLU.

        Raises:
            ValueError: If the image values are not provided in the configuration, or if the output data type is empty.
        """
        assert configuration.image_values is not None
        assert configuration.vision_transformer_config is None
        classes = configuration.output_data_type.values()
        assert classes is not None
        n_classes = len(classes)
        self.data_types = configuration.input_data_types
        self.category_type = configuration.output_data_type
        self.model = SimpleCNN(n_classes, activation=activation)

    def embed(self: Self, x: Tensor | DataloaderT, **kwargs: Any) -> Any:
        """
        Embed data from a model

        Args:
            x: Input
            kwargs: Embedding Arguments

        Returns:
            Any
        """
        from torch import no_grad
        from numpy import concatenate

        with no_grad():
            if type(x) == Tensor:
                if len(x.shape) == 3:
                    x = x[None, ...]
                x = x.to(device=self.device, dtype=self.dtype)
            else:
                values = []
                labels = []
                for data in x:
                    _x = data[0]
                    _y = data[1]
                    values.append(self.embed(_x, **kwargs)[0])
                    labels.append(_y.argmax(-1).cpu().numpy())
                return concatenate(values), concatenate(labels)
            embedding = self.model.features(x)  # pyright: ignore
            embedding = embedding.reshape(-1, embedding.shape[1]).cpu().numpy()
            return embedding, None

    def visualize(self: Self, x: Any, **kwargs: Any) -> Any:
        """
        Visualize a model

        Args:
            x: Input
            kwargs: Visualization Arguments

        Returns:
            Data
        """
        from torch import no_grad
        from numpy import concatenate

        with no_grad():
            if type(x) == Tensor:
                if len(x.shape) == 3:
                    x = x[None, ...]
                x = x.to(device=self.device, dtype=self.dtype)
            else:
                dataes = []
                values = []
                labels = []
                for data in x:
                    _x = data[0]
                    _y = data[1]
                    out = self.visualize(_x, **kwargs)
                    dataes.append(out[0])
                    values.append(out[1])
                    labels.append(_y.argmax(-1).cpu().numpy())
                return concatenate(dataes), concatenate(values), concatenate(labels)
            n_levels = 1
            if "n_levels" in kwargs.keys():
                n_levels = kwargs["n_levels"]
                assert isinstance(n_levels, int)
                n_levels = (
                    n_levels
                    if n_levels < len(self.model.features)  # pyright: ignore
                    else len(self.model.features) - 1  # pyright: ignore
                )
            z = x
            for level in range(n_levels):
                z = self.model.features[level](z)  # pyright: ignore
            visual = z
            _x = x.cpu().numpy().transpose(0, 2, 3, 1)
            visual = visual.mean(1)
            visual = visual.cpu().numpy()
            return _x, visual, None

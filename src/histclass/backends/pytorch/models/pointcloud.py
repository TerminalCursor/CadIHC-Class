from ....interfaces.config import CompleteConfig
from ....interfaces.data.types import DataTypes, CategoryType
from ....interfaces.models.types import ModelType
from ....interfaces.models.types import LossT, OptimizerT
from .interface import PyTorchModel
from ..types import ActivationT
from typing import Self, List
from torch.nn import Module, ModuleList, Sequential
from torch import Tensor
from torch.nn.functional import elu
from torch import nn


class SimpleHiddenLayer(Module):
    """
    # Simple Hidden Layer

    This layer applies a linear transformation followed by activation and dropout.

    Parameters:
    dim_in (int): The number of input features.
    dim_out (int): The number of output features.
    dropout (float, optional): The dropout probability. Defaults to 0.1.
    activation (ActivationT, optional): The activation function. Defaults to elu.
    """

    def __init__(
        self: Self,
        dim_in: int,
        dim_out: int,
        dropout: float = 0.1,
        activation: ActivationT = elu,
    ):
        """# Initialize Simple Hidden Layer"""
        super(SimpleHiddenLayer, self).__init__()
        self.layer = nn.Linear(dim_in, dim_out)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self: Self, x: Tensor) -> Tensor:
        """
        # Forward pass of simple hidden layer

        Parameters:
        x (Tensor): The input tensor.

        Returns:
        Tensor: The output tensor.
        """
        x = self.layer(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class AttentionBlock(Module):
    """
    # Attention Block

    This block applies multi-head attention followed by feedforward network.

    Parameters:
    embed_dim (int): The embedding dimension.
    num_heads (int): The number of attention heads.
    dim_feedforward (int): The feedforward network dimension.
    dropout (float, optional): The dropout probability. Defaults to 0.1.
    activation (ActivationT, optional): The activation function. Defaults to elu.
    """

    def __init__(
        self: Self,
        embed_dim: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: ActivationT = elu,
    ):
        """# Initialize Attention Block"""
        super(AttentionBlock, self).__init__()
        self.attention_layer = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(embed_dim)
        self.feedforward_layer = nn.Linear(embed_dim, dim_feedforward)
        self.feedforward_dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(dim_feedforward, embed_dim)
        self.output_dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self: Self, x: Tensor) -> Tensor:
        """
        # Forward pass

        Parameters:
        x (Tensor): The input tensor.

        Returns:
        Tensor: The output tensor.
        """
        dx, _ = self.attention_layer(x, x, x)
        X_hat = x + dx
        X_norm = self.attention_norm(X_hat)

        z = self.feedforward_dropout(self.activation(self.feedforward_layer(X_norm)))
        z = self.output_dropout(self.activation(self.output_layer(z)))
        return z


class CloudFlatten(Module):
    """
    # Flatten Point Cloud

    This layer flattens the last two dimensions of the input tensor.

    Parameters:
    None
    """

    def __init__(self: Self):
        """# Cloud Flatten Init"""
        super(CloudFlatten, self).__init__()

    def forward(self: Self, x: Tensor) -> Tensor:
        """
        # Flatten Last Two Dimensions

        Parameters:
        x (Tensor): The input tensor.

        Returns:
        Tensor: The flattened tensor.
        """
        x_shape = (*x.shape[:-2], x.shape[-2] * x.shape[-1])
        return x.reshape(x_shape)


class CloudFill(Module):
    """
    # Pad Cloud

    This layer pads the input tensor to the desired size.

    Parameters:
    max_points (int): The maximum number of points.
    min_points (int, optional): The minimum number of points. Defaults to 0.
    """

    def __init__(self: Self, max_points: int, min_points: int = 0):
        """# Initialization

        max_points: The desired size of the point cloud
        min_points: The minimal size of the starting point cloud"""
        super(CloudFill, self).__init__()
        self.max_points = max_points
        self.min_points = min_points

    def forward(self: Self, x: Tensor) -> Tensor:
        """
        # Reshape given cloud to desired length

        Parameters:
        x (Tensor): The input tensor.

        Returns:
        Tensor: The reshaped tensor.
        """
        from torch import cat, zeros

        # Ensure cloud size is maintained
        assert (
            x.shape[-2] >= self.min_points
        ), f"Cloud must at least be of size {self.min_points}"
        assert x.shape[-2] <= self.max_points, "Point Count Exceeded"
        # Quick return for correctly sized clouds
        if x.shape[-2] == self.max_points:
            return x
        # Otherwise pad with zeros
        z_shape = (*x.shape[:-2], self.max_points - x.shape[-2], x.shape[-1])
        x = cat((x, zeros(*z_shape).to(x.device)), dim=-2)
        return x


class PyTorchPointcloudModel(Module):
    """
    # PyTorch Pointcloud Model

    This model applies attention block followed by classification layer.

    Parameters:
    configuration (CompleteConfig): The configuration.
    activation (ActivationT, optional): The activation function. Defaults to elu.
    """

    embed_dim: int
    max_points: int
    num_head: int
    min_points: int
    n_blocks: int
    n_ff_nodes: int
    n_ff_layers: int
    dropout: float
    dropout_output: bool

    def __init__(
        self: Self,
        configuration: CompleteConfig,
        activation: ActivationT = elu,
    ):
        """# PyTorch Pointcloud Model"""
        super(PyTorchPointcloudModel, self).__init__()
        assert configuration.pointcloud_config is not None
        self.embed_dim = configuration.input_data_types.dimensionality()
        classes = configuration.output_data_type.values()
        assert classes is not None
        self.n_classes = len(classes)
        self.max_points = configuration.pointcloud_config.max_points
        self.min_points = configuration.pointcloud_config.min_points
        self.num_head = self.embed_dim
        self.n_blocks = configuration.pointcloud_config.n_blocks
        self.n_ff_nodes = configuration.pointcloud_config.n_ff_nodes
        self.n_ff_layers = configuration.pointcloud_config.n_ff_layers
        self.dropout = configuration.pointcloud_config.dropout
        self.dropout_output = configuration.pointcloud_config.dropout_output
        self.activation = activation
        self.cloud_fill = CloudFill(self.max_points, self.min_points)
        self.attention_layers = ModuleList(
            [
                AttentionBlock(
                    self.embed_dim,
                    self.num_head,
                    self.n_ff_nodes,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for _ in range(self.n_blocks)
            ]
        )
        self.cloud_flatten = CloudFlatten()
        classifier_layers: List[Module] = []
        assert isinstance(self.n_ff_layers, int)
        assert self.n_ff_layers > 0
        if self.n_ff_layers == 1:
            classifier_layers.append(
                SimpleHiddenLayer(
                    self.embed_dim * self.max_points,
                    self.n_classes,
                    dropout=self.dropout if self.dropout_output else 0,
                    activation=lambda x: x,
                )
            )
        else:
            classifier_layers.append(
                SimpleHiddenLayer(
                    self.embed_dim * self.max_points,
                    self.n_ff_nodes,
                    dropout=self.dropout,
                )
            )
            for _ in range(self.n_ff_layers - 2):
                classifier_layers.append(
                    SimpleHiddenLayer(
                        self.n_ff_nodes,
                        self.n_ff_nodes,
                        dropout=self.dropout,
                    )
                )
            classifier_layers.append(
                SimpleHiddenLayer(
                    self.n_ff_nodes,
                    self.n_classes,
                    dropout=self.dropout if self.dropout_output else 0,
                    activation=lambda x: x,
                )
            )
        self.classifier = Sequential(*classifier_layers)
        self.transfer = nn.functional.sigmoid

    def forward(self: Self, x: Tensor) -> Tensor:
        """
        # Procedure for Forward Pass

        Parameters:
        x (Tensor): The input tensor.

        Returns:
        Tensor: The output tensor.
        """
        x = self.cloud_fill(x)
        for attention_layer in self.attention_layers:
            x = attention_layer(x)
        X_t = self.cloud_flatten(x)
        classified = self.classifier(X_t)
        return self.transfer(classified)


class PointcloudModel(PyTorchModel):
    """
    # Pointcloud Model

    This model inherits from PyTorchPointcloudModel.
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
        """# Model Initialization"""
        self.data_types = configuration.input_data_types
        self.category_type = configuration.output_data_type
        self.model = PyTorchPointcloudModel(configuration, activation)

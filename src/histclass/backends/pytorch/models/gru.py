from ....interfaces.config import CompleteConfig
from ....interfaces.data.types import DataTypes, CategoryType
from ....interfaces.models.types import ModelType
from ....interfaces.models.types import LossT, OptimizerT
from .interface import PyTorchModel
from ..types import ActivationT
from typing import Self
from torch.nn import Module
from torch import Tensor
from torch.nn.functional import elu
from torch import nn


class PyTorchGru(Module):
    def __init__(
        self: Self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # input shape: (batch, seq_len, input_size)
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self: Self, x: Tensor) -> Tensor:
        """
        x shape: (batch_size, seq_len, input_size)
        """
        from torch import zeros

        # Initialize hidden state
        h0 = zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)

        # RNN forward pass
        out, _ = self.rnn(x, h0)

        # Take last timestep output
        out = out[:, -1, :]  # (batch_size, hidden_size)

        # Final classification layer
        out = self.fc(out)  # (batch_size, num_classes)

        return out


class GruModel(PyTorchModel):
    """# GRU Model"""

    model: Module
    optimizer: OptimizerT
    loss_fn: LossT

    model_type: ModelType = ModelType.Recurrent
    data_types: DataTypes
    category_type: CategoryType

    def __init__(
        self: Self,
        configuration: CompleteConfig,
        activation: ActivationT = elu,
    ):
        """# Model Initialization"""
        assert configuration.pointcloud_config is not None
        hidden_size = configuration.pointcloud_config.n_ff_nodes
        num_layers = configuration.pointcloud_config.n_ff_layers
        dropout = configuration.pointcloud_config.dropout
        self.data_types = configuration.input_data_types
        self.category_type = configuration.output_data_type
        embed_dim = self.data_types.dimensionality()
        classes = self.category_type.values()
        assert classes is not None
        n_classes = len(classes)
        self.model = PyTorchGru(
            embed_dim, hidden_size, num_layers, n_classes, dropout=dropout
        )

from ....tools.call_graph import cg
from ....interfaces.config import CompleteConfig
from ....interfaces.data.dataset import Dataset
from .interface import PyTorchModel
from ..data.dataset import PointcloudDataframe
from typing import Self, Callable
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
        cg.add("PyTorchGRU.__init__")
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
        cg.add("PyTorchGRU.forward")
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
    optimizer: Callable
    loss_fn: Callable

    def __init__(
        self: Self,
        dataset: Dataset,
        configuration: CompleteConfig,
        activation: Callable = elu,
    ):
        """# Model Initialization"""
        cg.add("GRUModel.__init__")
        assert isinstance(dataset.data_frames[0], PointcloudDataframe)
        assert configuration.pointcloud_config is not None
        embed_dim = dataset.data_frames[0].data.shape[-1]
        hidden_size = configuration.pointcloud_config.n_ff_nodes
        num_layers = configuration.pointcloud_config.n_ff_layers
        n_classes = dataset.data_frames[0].labels.shape[-1]
        dropout = configuration.pointcloud_config.dropout
        self.model = PyTorchGru(
            embed_dim, hidden_size, num_layers, n_classes, dropout=dropout
        )

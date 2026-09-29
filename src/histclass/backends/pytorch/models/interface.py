from ....tools.call_graph import cg
from ....interfaces.models.model import Model
from ....interfaces.config import CompleteConfig
from ....interfaces.data.types import DataTypes, CategoryType
from ....interfaces.models.types import ModelType
from ....interfaces.models.types import LossT, OptimizerT
from ..types import DataloaderT
from typing import Self, Tuple, Any
from torch.nn import Module
from torch import Tensor
from torch.nn.functional import elu
from torch.utils.data import DataLoader
from torch import float32
from torch import dtype as _DTYPE
from numpy.typing import NDArray


def accuracy_from_matrix(matrix: Tensor) -> float:
    """# Get Accuracy From Matrix"""
    cg.add("accuracy_from_matrix")
    from torch import eye

    correct = (matrix * eye(matrix.shape[0])).sum().item()
    total = matrix.sum().item()
    return correct / total


class PyTorchModel(Model):
    """# PyTorch Model Interface"""

    device: str = "cpu"
    dtype: _DTYPE = float32
    model: Module
    optimizer: OptimizerT
    loss_fn: LossT

    model_type: ModelType = ModelType.Attention
    data_types: DataTypes
    category_type: CategoryType

    def __init__(
        self: Self,
        configuration: CompleteConfig,
        activation: LossT = elu,
    ):
        """# Model Initialization"""
        cg.add("PyTorchModel.__init__")
        raise NotImplementedError("Implement me in subclasses!")

    def set_optimizer(self: Self, optimizer: OptimizerT) -> Self:
        """# Optimizer for Model Training"""
        cg.add("PyTorchModel.set_optimizer")
        self.optimizer = optimizer
        return self

    def set_loss_fn(self: Self, loss_fn: LossT) -> Self:
        """# Loss Function for Model Training"""
        cg.add("PyTorchModel.set_loss_fn")
        self.loss_fn = loss_fn
        return self

    def set_device(self: Self, device: Any | None) -> Self:
        """# Device to put Model On"""
        cg.add("PyTorchModel.set_device")
        if device is not None:
            self.device = device
        self.model = self.model.to(self.device)
        return self

    def set_dtype(self: Self, dtype: Any | None) -> Self:
        """# Dtype for Model to use"""
        cg.add("PyTorchModel.set_dtype")
        if dtype is not None:
            self.dtype = dtype
        self.model = self.model.to(self.dtype)
        return self

    def train_batch(
        self: Self,
        dataloader: DataloaderT,
        loss_fn: LossT,
    ) -> Tuple[float, float, NDArray[Any]]:
        """# Procedure to Train on Batch"""
        cg.add("PyTorchModel.train_batch")
        from torch import zeros, int32
        from torch.nn.functional import softmax

        self.model.train()
        total_loss = 0.0
        n_loss = 0
        n_class: int = dataloader.dataset[0][1].shape[-1]
        acc_matrix = zeros((n_class, n_class), dtype=int32)
        for data, label in dataloader:
            self.model.zero_grad()
            data = data.to(self.device, self.dtype)
            y_true = label.to(self.device, self.dtype)
            forward = self.model(data)
            y_pred = softmax(forward, dim=-1)
            loss = loss_fn(y_pred, y_true)
            loss.backward()
            self.optimizer.step()  # pyright: ignore
            total_loss += loss.item()
            n_loss += 1
            for actual, pred in zip(y_true, y_pred):
                actual = int(actual.argmax().item())
                pred = int(pred.argmax().item())
                acc_matrix[actual, pred] += 1
        acc = accuracy_from_matrix(acc_matrix)
        acc_matrix = acc_matrix.detach().cpu().numpy()
        return total_loss / n_loss, acc, acc_matrix

    def eval_batch(
        self: Self,
        dataloader: DataLoader[Tuple[Tensor, Tensor]],
        loss_fn: LossT,
    ) -> Tuple[float, float, NDArray[Any]]:
        """# Procedure to Evaluate on Batch"""
        cg.add("PyTorchModel.eval_batch")
        from torch import no_grad, zeros, int32
        from torch.nn.functional import softmax

        self.model.eval()
        total_loss = 0.0
        n_loss = 0
        n_class = dataloader.dataset[0][1].shape[-1]
        acc_matrix = zeros((n_class, n_class), dtype=int32)
        with no_grad():
            for data, label in dataloader:
                data = data.to(self.device, self.dtype)
                y_true = label.to(self.device, self.dtype)
                forward = self.model(data)
                y_pred = softmax(forward, dim=-1)
                loss = loss_fn(y_pred, y_true)
                total_loss += loss.item()
                n_loss += 1
                for actual, pred in zip(y_true, y_pred):
                    actual = int(actual.argmax().item())
                    pred = int(pred.argmax().item())
                    acc_matrix[actual, pred] += 1
        acc = accuracy_from_matrix(acc_matrix)
        acc_matrix = acc_matrix.detach().cpu().numpy()
        return total_loss / n_loss, acc, acc_matrix

    def to(self: Self, device: Any | None = None, dtype: Any | None = None) -> Self:
        """# Move Model to Device"""
        cg.add("PyTorchModel.to")
        self.set_device(device)
        self.set_dtype(dtype)
        return self

    def load(self: Self, checkpoint_path: str) -> Self:
        """# Load Model from Checkpoint"""
        cg.add("PyTorchModel.load")
        from torch import load

        checkpoint = load(checkpoint_path)
        self.model.load_state_dict(checkpoint)
        return self

    def save(self: Self, checkpoint_path: str) -> Self:
        """# Save Model to Checkpoint"""
        cg.add("PyTorchModel.save")
        from torch import save

        save(self.model.state_dict(), checkpoint_path)
        return self

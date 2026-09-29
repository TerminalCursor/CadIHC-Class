from ....interfaces.models.model import Model
from ....interfaces.config import CompleteConfig
from ....interfaces.data.types import DataTypes, CategoryType
from ....interfaces.models.types import ModelType
from ....interfaces.models.types import LossT, OptimizerT
from ..types import DataloaderT
from typing import Self, Tuple, List, Any
from torch.nn import Module
from torch import Tensor
from torch.nn.functional import elu
from torch import float32
from torch import dtype as _DTYPE
from numpy.typing import NDArray


def accuracy_from_matrix(matrix: Tensor) -> float:
    """
    Compute the accuracy from a matrix.

    Args:
        matrix (Tensor): A square tensor representing the confusion matrix.

    Returns:
        float: The accuracy value.
    """
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

    classic_data_norm: bool = False
    confidences: List[float] | None = None
    accuracy: float | None = None

    def __init__(
        self: Self,
        configuration: CompleteConfig,
        activation: LossT = elu,
    ):
        """
        Initialize the model.

        Args:
            configuration (CompleteConfig): The model configuration.
            activation (LossT, optional): The loss function activation. Defaults to elu.
        """
        raise NotImplementedError("Implement me in subclasses!")

    def set_optimizer(self: Self, optimizer: OptimizerT) -> Self:
        """
        Set the optimizer for the model.

        Args:
            optimizer (OptimizerT): The optimizer to use.

        Returns:
            Self
        """
        self.optimizer = optimizer
        return self

    def set_loss_fn(self: Self, loss_fn: LossT) -> Self:
        """
        Set the loss function for the model.

        Args:
            loss_fn (LossT): The loss function to use.

        Returns:
            Self
        """
        self.loss_fn = loss_fn
        return self

    def set_device(self: Self, device: Any | None) -> Self:
        """
        Set the device to put the model on.

        Args:
            device (Any | None): The device to use. Defaults to None.

        Returns:
            Self
        """
        if device is not None:
            self.device = device
        self.model = self.model.to(self.device)
        return self

    def set_dtype(self: Self, dtype: Any | None) -> Self:
        """
        Set the data type for the model.

        Args:
            dtype (Any | None): The data type to use. Defaults to None.

        Returns:
            Self
        """
        if dtype is not None:
            self.dtype = dtype
        self.model = self.model.to(self.dtype)
        return self

    def train_epoch(
        self: Self,
        dataloader: DataloaderT,
        loss_fn: LossT,
    ) -> Tuple[float, float, NDArray[Any]]:
        """
        Train the model on a batch.

        Args:
            dataloader (DataloaderT): The data loader to use.
            loss_fn (LossT): The loss function to use.

        Returns:
            Tuple[float, float, NDArray[Any]]: A tuple containing the total loss, accuracy,
                and the confusion matrix.
        """
        from torch import zeros, int32
        from torch.nn.functional import softmax

        self.model.train()
        total_loss = 0.0
        n_loss = 0
        n_class: int = dataloader.dataset[0][1].shape[-1]
        acc_matrix = zeros((n_class, n_class), dtype=int32)
        for data, label, _ in dataloader:
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

    def eval_epoch(
        self: Self,
        dataloader: DataloaderT,
        loss_fn: LossT,
    ) -> Tuple[float, float, NDArray[Any]]:
        """
        Evaluate the model on a batch.

        Args:
            dataloader (DataloaderT): The data loader to use.
            loss_fn (LossT): The loss function to use.

        Returns:
            Tuple[float, float, NDArray[Any]]: A tuple containing the total loss, accuracy,
                and the confusion matrix.
        """
        from torch import no_grad, zeros, int32
        from torch.nn.functional import softmax

        self.model.eval()
        total_loss = 0.0
        n_loss = 0
        n_class = dataloader.dataset[0][1].shape[-1]
        acc_matrix = zeros((n_class, n_class), dtype=int32)
        with no_grad():
            for data, label, _ in dataloader:
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
        """
        Move the model to a specific device.

        Args:
            device (Any | None): The device to move to. Defaults to None.
            dtype (Any | None): The data type to use. Defaults to None.

        Returns:
            Self
        """
        self.set_device(device)
        self.set_dtype(dtype)
        return self

    def load(self: Self, checkpoint_path: str) -> Self:
        """
        Load a model from a checkpoint file.

        Args:
            checkpoint_path (str): The path to the checkpoint file.

        Returns:
            Self
        """
        from torch import load

        checkpoint = load(checkpoint_path)
        self.model.load_state_dict(checkpoint)
        return self

    def save(self: Self, checkpoint_path: str) -> Self:
        """
        Save a model to a checkpoint file.

        Args:
            checkpoint_path (str): The path to the checkpoint file.

        Returns:
            Self
        """
        from torch import save

        save(self.model.state_dict(), checkpoint_path)
        return self

    def embed(self: Self, x: Any, **kwargs: Any) -> Any:
        """
        Embed data from a model

        Args:
            x: Input
            kwargs: Embedding Arguments

        Returns:
            Data
        """
        return None

    def visualize(self: Self, x: Any, **kwargs: Any) -> Any:
        """
        Visualize a model

        Args:
            x: Input
            kwargs: Visualization Arguments

        Returns:
            Data
        """
        return None

from ...tools.call_graph import cg
from ...interfaces.config import CompleteConfig
from typing import Callable
from torch.nn.functional import mse_loss
from torch.nn.functional import cross_entropy


from torch import Tensor, log, sigmoid
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Self


class FocalLoss(nn.Module):
    """# Focal Loss

    From: https://github.com/itakurah/focal-loss-pytorch
    """

    def __init__(
        self: Self,
        gamma: float = 2,
        alpha: Tensor | None = None,
        reduction: str = "mean",
        task_type: str = "binary",
        num_classes: int | None = None,
    ):
        """
        Unified Focal Loss class for binary, multi-class, and multi-label classification tasks.
        :param gamma: Focusing parameter, controls the strength of the modulating factor (1 - p_t)^gamma
        :param alpha: Balancing factor, can be a scalar or a tensor for class-wise weights. If None, no class balancing is used.
        :param reduction: Specifies the reduction method: 'none' | 'mean' | 'sum'
        :param task_type: Specifies the type of task: 'binary', 'multi-class', or 'multi-label'
        :param num_classes: Number of classes (only required for multi-class classification)
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.task_type = task_type
        self.num_classes = num_classes

        # Handle alpha for class balancing in multi-class tasks
        if (
            task_type == "multi-class"
            and alpha is not None
            and isinstance(alpha, (list, Tensor))
        ):
            assert (
                num_classes is not None
            ), "num_classes must be specified for multi-class classification"
            if isinstance(alpha, list):
                self.alpha = Tensor(alpha)
            else:
                self.alpha = alpha

    def forward(self: Self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Forward pass to compute the Focal Loss based on the specified task type.
        :param inputs: Predictions (logits) from the model.
                       Shape:
                         - binary/multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size, num_classes)
        :param targets: Ground truth labels.
                        Shape:
                         - binary: (batch_size,)
                         - multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size,)
        """
        if self.task_type == "binary":
            return self.binary_focal_loss(inputs, targets)
        elif self.task_type == "multi-class":
            return self.multi_class_focal_loss(inputs, targets)
        elif self.task_type == "multi-label":
            return self.multi_label_focal_loss(inputs, targets)
        else:
            raise ValueError(
                f"Unsupported task_type '{self.task_type}'. Use 'binary', 'multi-class', or 'multi-label'."
            )

    def binary_focal_loss(self: Self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Focal loss for binary classification."""
        probs = sigmoid(inputs)
        targets = targets.float()

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weighting
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    def multi_class_focal_loss(self: Self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Focal loss for multi-class classification."""
        assert self.num_classes is not None
        if self.alpha is not None:
            self.alpha = self.alpha.to(inputs.device)

        # Convert logits to probabilities with softmax
        probs = F.softmax(inputs, dim=1)

        # One-hot encode the targets
        if targets.shape[-1] == 1:
            targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        else:
            targets_one_hot = targets.float()

        # Compute cross-entropy for each class
        ce_loss = -targets_one_hot * log(probs)

        # Compute focal weight
        p_t = torch.sum(probs * targets_one_hot, dim=1)  # p_t for each sample
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided (per-class weighting)
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            ce_loss = alpha_t.unsqueeze(1) * ce_loss

        # Apply focal loss weight
        loss = focal_weight.unsqueeze(1) * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    def multi_label_focal_loss(self: Self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Focal loss for multi-label classification."""
        probs = sigmoid(inputs)

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weight
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def get_loss(config: CompleteConfig) -> Callable | None:
    """# Returns the loss function corresponding to the name

    Backend-specific function
    """
    cg.add("get_loss")
    name = config.loss_fn
    if name == "mse_loss":
        return mse_loss
    elif name == "cross_entropy":
        return cross_entropy
    elif name == "focal_loss":
        classes = config.output_data_type.values()
        assert classes is not None
        task_type = "binary" if len(classes) == 2 else "multi-class"
        return FocalLoss(task_type=task_type, num_classes=len(classes))
    return None

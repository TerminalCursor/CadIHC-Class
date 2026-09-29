from ...tools.call_graph import cg
from enum import Enum
from typing import Self, Callable, Any, Protocol


class OptimizerT(Protocol):
    """# Optimizer Protocol"""

    def __call__(self: Self, *args: Any, **kwargs: Any) -> Any:
        pass


LossT = Callable[[Any, Any], Any]


class ModelType(Enum):
    """# Model Type"""

    Attention = 0
    Recurrent = 1

    def to_str(self: Self) -> str:
        """# Model Type to String"""
        cg.add("ModelType.to_str")
        if self == ModelType.Attention:
            return "attention"
        elif self == ModelType.Recurrent:
            return "recurrent"

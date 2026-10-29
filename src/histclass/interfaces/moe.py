from typing import Any, Self, Tuple, List, Dict
from numpy.typing import NDArray


class MoEData:
    data: Any
    n_classes: int

    def __init__(self: Self, raw: Any, n_classes: int = 2):
        self.data = raw
        self.n_classes = n_classes

    def data_from_label(
        self: Self, model: Any | None = None, label: Any | None = None
    ) -> Tuple[List[Any], int, Any] | None:
        # Return [List of (Model, Probability), label, Data Name]
        raise NotImplementedError("Implement in Subclasses!")

    def sample(
        self: Self, model: Any | None = None, label: Any | None = None
    ) -> Tuple[NDArray[Any], List[Any], int, Any] | None:
        raise NotImplementedError("Implement in Subclasses!")


class MoEModels:
    model_confidences: Any
    model_accuracies: Any
    n_classes: int
    name_mapping: Dict[Any, int]

    def __init__(
        self: Self,
        raw: Any,
        acc: Any = None,
        name_mapping: Dict[Any, int] | None = None,
        n_classes: int = 2,
    ):
        raise NotImplementedError("Implement me in subclasses!")

    def get(self: Self, model: Any | None = None) -> NDArray[Any]:
        raise NotImplementedError("Implement me in subclasses!")

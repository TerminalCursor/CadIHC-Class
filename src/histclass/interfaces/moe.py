from typing import Any, Self, Tuple
from numpy.typing import NDArray


class MoEData:
    data: Any
    n_classes: int

    def __init__(self: Self, raw: Any, n_classes: int = 2):
        self.data = raw
        self.n_classes = n_classes

    def sample(
        self: Self, model: Any = None, label: Any = None, one: bool = True
    ) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any]] | None:
        raise NotImplementedError("Implement in Subclasses!")


class MoEModels:
    model_confidences: Any
    model_accuracies: Any
    n_classes: int

    def __init__(self: Self, raw: Any, acc: Any = None, n_classes: int = 2):
        raise NotImplementedError("Implement me in subclasses!")

    def get(self: Self, model: Any = None) -> NDArray[Any]:
        raise NotImplementedError("Implement me in subclasses!")

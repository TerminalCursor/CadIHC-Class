from typing import Self, Protocol, Any


class FuncT(Protocol):
    """# Generic Function Protocol"""

    def __call__(self: Self, *args: Any, **kwargs: Any) -> Any:
        pass

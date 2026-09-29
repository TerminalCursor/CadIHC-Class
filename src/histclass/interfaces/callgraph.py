from typing import List, Self, Any
from .types import FuncT


class Callgraph:
    """# Callgraph Interface

    add: Add data to callgraph
    clear: Reset callgraph
    save: Save callgraph
    toggle: Toggle if callgraph should record
    __call__: Allow calling functions with callgraph enabled
    fun_follow: Save total function callgraph to `name`
    """

    data: List[str] = []
    enabled: bool = False

    def __init__(self: Self):
        """# Initialize Callgraph"""
        self.data = []
        self.enabled = False

    def add(self: Self, data: str) -> Self:
        """# Add Data To Callgraph"""
        raise NotImplementedError("Implement me!")

    def clear(self: Self) -> Self:
        """# Clear Callgraph"""
        raise NotImplementedError("Implement me!")

    def save(self: Self, *args: Any, **kwargs: Any) -> Self:
        """# Save Callgraph"""
        raise NotImplementedError("Implement me!")

    def toggle(self: Self, to: bool | None = None) -> Self:
        """# Toggle Callgraph Monitoring"""
        if to is not None:
            self.enabled = to
            return self
        self.enabled = not self.enabled
        return self

    def __call__(self: Self, fptr: FuncT, *args: Any, **kwargs: Any) -> Any:
        """# Call Function With Callgraph"""
        original_state = self.enabled
        self.toggle(to=True)
        output = fptr(*args, **kwargs)
        self.toggle(to=original_state)
        return output

    def func_follow(
        self: Self, name: str, fptr: FuncT, *args: Any, **kwargs: Any
    ) -> Any:
        """# Call Function With Callgraph"""
        original_state = self.enabled
        self.toggle(to=True)
        self.clear()
        output = fptr(*args, **kwargs)
        self.save(name)
        self.clear()
        self.toggle(to=original_state)
        return output

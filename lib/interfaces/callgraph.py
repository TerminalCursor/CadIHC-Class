from typing import List, Self, Callable, Any


class Callgraph:
    data: List[str] = []
    enabled: bool = False

    def add(self, data: str) -> Self:
        """# Add Data To Callgraph"""
        raise NotImplementedError("Implement me!")

    def clear(self) -> Self:
        """# Clear Callgraph"""
        raise NotImplementedError("Implement me!")

    def save(self, *args, **kwargs) -> Self:
        """# Save Callgraph"""
        raise NotImplementedError("Implement me!")

    def toggle(self, to: bool | None = None) -> Self:
        """# Toggle Callgraph Monitoring"""
        if to is not None:
            self.enabled = to
            return self
        self.enabled = not self.enabled
        return self

    def __call__(self, fptr: Callable, *args, **kwargs) -> Any:
        """# Call Function With Callgraph"""
        original_state = self.enabled
        self.toggle(to=True)
        output = fptr(*args, **kwargs)
        self.toggle(to=original_state)
        return output

    def func_follow(self, name: str, fptr: Callable, *args, **kwargs) -> Any:
        """# Call Function With Callgraph"""
        original_state = self.enabled
        self.toggle(to=True)
        self.clear()
        output = fptr(*args, **kwargs)
        self.save(name)
        self.clear()
        self.toggle(to=original_state)
        return output

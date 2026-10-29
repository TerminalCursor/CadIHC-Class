from typing import Self, Protocol, Any


class FuncT(Protocol):
    """
    Generic Function Protocol Interface.

    This interface defines the signature of a generic function that can be used with various types of functions.

    Attributes:
        None

    Methods:
        __call__: The entry point of the function. It must accept any number of positional and keyword arguments,
                  and return a value of the same type as the input arguments.
    """

    def __call__(self: Self, *args: Any, **kwargs: Any) -> Any:
        """
        Entry point of the function.

        This method is called when an instance of `FuncT` is invoked like a function. It must accept any number
        of positional and keyword arguments, and return a value of the same type as the input arguments.

        Args:
            self (Self): The instance of `FuncT`.
            *args: Any number of positional arguments.
            **kwargs: Any number of keyword arguments.

        Returns:
            Any: A value of the same type as the input arguments.

        Raises:
            NotImplementedError: If a subclass of `FuncT` does not implement this method.
        """
        pass

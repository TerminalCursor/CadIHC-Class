from ..tools.call_graph import cg
from typing import Self, Any


class Database:
    """# Database Class

    Handles a SQLite3 Database
    """

    filename: str

    def __init__(self: Self, filename: str):
        """# Initialization

        Associate with filename
        """
        cg.add("Database.__init__")
        self.filename = filename

    def add(self: Self, *_args: Any, **_results: Any) -> Self:
        """# Method to add data to database"""
        raise NotImplementedError("Implement me!")

    def write(self: Self, *_args: Any, **_results: Any) -> Self:
        """# Method to write added data to database on disk"""
        raise NotImplementedError("Implement me!")

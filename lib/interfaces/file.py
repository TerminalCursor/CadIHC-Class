from ..tools.call_graph import cg
from typing import Self


class File:
    """# File Class

    Handles the creation and appending of data to a file
    """

    def __init__(self: Self, filename: str):
        """# Set the filename to manage"""
        cg.add("File.__init__")
        self.filename = filename

    def create(self: Self) -> Self:
        """# Create the file"""
        cg.add("File.create")
        with open(self.filename, "w+") as fhandle:
            fhandle.write("")
        return self

    def append(self: Self, stream: str) -> Self:
        """# Append data to the file"""
        cg.add("File.append")
        with open(self.filename, "a") as fhandle:
            fhandle.write(stream)
        return self

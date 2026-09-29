from typing import Self


class File:
    """# File Class

    Handles the creation and appending of data to a file
    """

    filename: str

    def __init__(self: Self, filename: str):
        """# Set the filename to manage"""
        self.filename = filename

    def create(self: Self) -> Self:
        """# Create the file"""
        with open(self.filename, "w+") as fhandle:
            fhandle.write("")
        return self

    def append(self: Self, stream: str) -> Self:
        """# Append data to the file"""
        with open(self.filename, "a") as fhandle:
            fhandle.write(stream)
        return self

    def read(self: Self) -> str:
        """# Append data to the file"""
        data: str = ""
        with open(self.filename, "r") as fhandle:
            data = fhandle.read()
        return data

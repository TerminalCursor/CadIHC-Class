from typing import Self


class File:
    """
    A class used to handle the creation and appending of data to a file.

    Attributes:
    ----------
    filename : str
        The name of the file being managed.
    """

    filename: str

    def __init__(self: Self, filename: str):
        """
        Initializes a new instance of the File class with the specified filename.

        Parameters:
        ----------
        filename : str
            The name of the file to be created or appended to.
        """
        self.filename = filename

    def create(self: Self) -> Self:
        """
        Creates a new file with the specified filename and writes an empty string to it.

        Returns:
        -------
        File
            The instance itself, allowing for method chaining.
        """
        with open(self.filename, "w+") as fhandle:
            fhandle.write("")
        return self

    def append(self: Self, stream: str) -> Self:
        """
        Appends the specified data stream to the end of the existing file.

        Parameters:
        ----------
        stream : str
            The data to be appended to the file.

        Returns:
        -------
        File
            The instance itself, allowing for method chaining.
        """
        with open(self.filename, "a") as fhandle:
            fhandle.write(stream)
        return self

    def read(self: Self) -> str:
        """
        Reads the entire contents of the file.

        Returns:
        -------
        str
            The contents of the file.
        """
        data: str = ""
        with open(self.filename, "r") as fhandle:
            data = fhandle.read()
        return data

from ..tools.call_graph import cg


class File:
    """# File Class

    Handles the creation and appending of data to a file
    """

    def __init__(self, filename: str):
        """# Set the filename to manage"""
        cg.add("File.__init__")
        self.filename = filename

    def create(self):
        """# Create the file"""
        cg.add("File.create")
        with open(self.filename, "w+") as fhandle:
            fhandle.write("")

    def append(self, stream: str):
        """# Append data to the file"""
        cg.add("File.append")
        with open(self.filename, "a") as fhandle:
            fhandle.write(stream)

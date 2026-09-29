from ..tools.call_graph import cg
from .db import Database
from typing import Self, Any


class Logger:
    """# Logger Class

    Enable logging into text dump and database

    `add` and `addnl` adds text to a logfile
    `db_set_meta` set metadata for database recording
    `db_add_data` adds data to database
    `db_update` updates the added data to the database on disk
    """

    enabled: bool = True
    echo_stdout: bool = True
    timestamp: str = ""
    db: Database

    def __init__(self: Self, base_path: str, echo_stdout: bool = True):
        """# Initialize Logger"""
        cg.add("Logger.__init__")
        from ..tools.date import get_datestring
        from os.path import exists
        from .file import File

        self.echo_stdout = echo_stdout

        self.timestamp = get_datestring()

        assert exists(base_path)

        self.file = File(f"{base_path}/{self.timestamp}.txt")
        self.file.create()

    def add(self: Self, formatted_text: dict | tuple | str) -> Self:
        """# Add Text to File

        If provided text is a dict or tuple, format it as a string with indentations
        """
        cg.add("Logger.add")
        if not self.enabled:
            return self

        if isinstance(formatted_text, dict):
            import json

            return self.add(json.dumps(formatted_text, indent=4))

        if isinstance(formatted_text, tuple):
            import json

            return self.add(json.dumps(formatted_text, indent=4))

        if self.echo_stdout:
            print(formatted_text, end="")
        self.file.append(formatted_text)
        return self

    def addnl(self: Self, formatted_text: dict | tuple | str) -> Self:
        """# Add Text with Newline

        See `add` for more details
        """
        cg.add("Logger.addnl")
        self.add(formatted_text)
        self.add("\n")
        return self

    def db_add(self: Self, **results_dict: Any) -> Self:
        """# Add Data to Database

        Database backend can set what keys will correspond to datapoints
        """
        cg.add("Logger.db_add")
        self.db.add(**results_dict)
        return self

    def db_write(self: Self) -> Self:
        """# Update DB with Data"""
        cg.add("Logger.db_write")
        self.db.write(echo=self.echo_stdout)
        return self

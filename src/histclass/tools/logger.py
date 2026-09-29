from .call_graph import cg
from typing import Self, Dict, Tuple, Any
from ..interfaces.logger import Logger as LoggerInterface
from ..interfaces.db import Database


class Logger(LoggerInterface):
    """# Logger Class

    Enable logging into text dump and database

    `add` and `addnl` adds text to a logfile
    `db_set_meta` set metadata for database recording
    `db_add_data` adds data to database
    `db_update` updates the added data to the database on disk
    """

    db: Database
    enabled: bool = True
    echo_stdout: bool = True
    timestamp: str = ""

    def __init__(self: Self, base_path: str, echo_stdout: bool = True):
        """# Initialize Logger"""
        cg.add("Logger.__init__")
        super().__init__(base_path, echo_stdout)
        from .results_db import ResultsDatabase

        self.db = ResultsDatabase(f"{base_path}/Results.db")
        self.db_add(timestamp=self.timestamp)

    def add(self: Self, formatted_text: Dict[Any, Any] | Tuple[Any, ...] | str) -> Self:
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

    def addnl(
        self: Self, formatted_text: Dict[Any, Any] | Tuple[Any, ...] | str
    ) -> Self:
        """# Add Text with Newline

        See `add` for more details
        """
        cg.add("Logger.addnl")
        self.add(formatted_text)
        self.add("\n")
        return self

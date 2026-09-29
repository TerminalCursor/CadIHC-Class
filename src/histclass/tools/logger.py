from typing import Self, Dict, Tuple, Any
from ..interfaces.logger import Logger as LoggerInterface
from ..interfaces.db import Database


class Logger(LoggerInterface):
    """
    # Logger Class

    The Logger class provides methods for logging text to a file and setting metadata for the database.

    Attributes:
        db (Database): An instance of the ResultsDatabase class, used for storing data in the database.
        enabled (bool): A flag indicating whether logging is enabled. Defaults to True.
        echo_stdout (bool): A flag indicating whether logging should be printed to the stdout. Defaults to True.
        timestamp (str): The current timestamp.
    """

    db: Database
    enabled: bool = True
    echo_stdout: bool = True
    timestamp: str = ""

    def __init__(self: Self, base_path: str, echo_stdout: bool = True):
        """
        # Initialize Logger

        Initializes the Logger instance with a base path and optionally enables stdout logging.

        Args:
            base_path (str): The base path for the logger.
            echo_stdout (bool): An optional flag to enable or disable stdout logging. Defaults to True.
        """
        super().__init__(base_path, echo_stdout)
        from .results_db import ResultsDatabase

        self.db = ResultsDatabase(f"{base_path}/Results.db")
        self.db_add(timestamp=self.timestamp)

    def add(self: Self, formatted_text: Dict[Any, Any] | Tuple[Any, ...] | str) -> Self:
        """
        # Add Text to File

        Adds text to the log file. If provided text is a dict or tuple, it will be formatted as a string with indentations.

        Args:
            formatted_text (Dict[Any, Any] | Tuple[Any, ...] | str): The text to add to the log file.

        Returns:
            Self: The updated Logger instance.
        """
        if not self.enabled:
            return self

        if isinstance(formatted_text, dict):
            from json import dumps

            return self.add(dumps(formatted_text, indent=4))

        if isinstance(formatted_text, tuple):
            from json import dumps

            return self.add(dumps(formatted_text, indent=4))

        if self.echo_stdout:
            print(formatted_text, end="")
        self.file.append(formatted_text)
        return self

    def addnl(
        self: Self, formatted_text: Dict[Any, Any] | Tuple[Any, ...] | str
    ) -> Self:
        """
        # Add Text with Newline

        Adds text to the log file followed by a newline character.

        Args:
            formatted_text (Dict[Any, Any] | Tuple[Any, ...] | str): The text to add to the log file.

        Returns:
            Self: The updated Logger instance.
        """
        self.add(formatted_text)
        self.add("\n")
        return self

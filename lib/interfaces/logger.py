from ..tools.call_graph import cg


class Logger:
    """# Logger Class

    Enable logging into text dump and database
    """

    enabled: bool = True
    echo_stdout: bool = True
    timestamp: str = ""

    def __init__(self, base_path: str, echo_stdout: bool = True):
        """# Initialize Logger"""
        cg.add("Logger.__init__")
        from ..tools.date import get_datestring
        from ..tools.results_db import ResultsDatabase
        from os.path import exists
        from .file import File

        self.echo_stdout = echo_stdout

        self.timestamp = get_datestring()

        assert exists(base_path)

        self.db = ResultsDatabase(f"{base_path}/Results.db")

        self.file = File(f"{base_path}/{self.timestamp}.txt")
        self.file.create()

    def add(self, formatted_text: dict | tuple | str):
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

    def addnl(self, formatted_text: dict | tuple | str):
        """# Add Text with Newline

        See `add` for more details
        """
        cg.add("Logger.addnl")
        self.add(formatted_text)
        self.add("\n")

    def db_set_meta(self, name: str, category: str, lr: float, loss: str):
        """# Set Results DB Metadata"""
        cg.add("Logger.set_db_meta")
        self.db.timestamp = self.timestamp
        self.db.name = name
        self.db.category = category
        self.db.learning_rate = lr
        self.db.loss = loss

    def db_add_data(self, **results_dict: float | str):
        """# Add Data to Dictionary

        Scalar Result Keys:
        total_accuracy, training_accuracy, validation_accuracy, total_accuracy_c, training_accuracy_c, validation_accuracy_c

        Matrix Result Keys:
        total_raw, training_raw, validation_raw
        """
        cg.add("Logger.db_add_data")
        self.db.add(**results_dict)

    def db_update_data(self):
        """# Update DB with Data"""
        cg.add("Logger.db_update_data")
        self.db.insert(echo=self.echo_stdout)

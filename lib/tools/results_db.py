from .call_graph import cg
from ..interfaces.db import Database
from typing import List


def sqlite3_execute(cursor, query, echo=False, dry_run=False):
    if echo:
        print(query)
    if not dry_run:
        cursor.execute(query)


class ResultsDatabase(Database):
    """# Results Database Class"""

    timestamp: str = ""
    name: str = ""
    category: str = ""
    learning_rate: float = 0.0
    loss: str = ""
    data: dict = {}

    scalar_result_keys: List[str] = [
        "total_accuracy",
        "training_accuracy",
        "validation_accuracy",
        "total_accuracy_c",
        "training_accuracy_c",
        "validation_accuracy_c",
    ]
    matrix_result_keys: List[str] = ["total_raw", "training_raw", "validation_raw"]

    def __init__(self, filename):
        """# Set Filename to Manage"""
        cg.add("ResultsDatabase.__init__")
        super().__init__(filename)

    def add(self, **results_dict):
        """# Add Data"""
        cg.add("ResultsDatabase.add")
        for key in results_dict.keys():
            keyl = key.lower()
            if keyl in self.scalar_result_keys:
                self.data[keyl] = float(results_dict[key])
            elif keyl in self.matrix_result_keys:
                self.data[keyl] = results_dict[key]

    def insert(self, echo=False):
        """# Insert Data

        From `self.data`
        """
        cg.add("ResultsDatabase.insert")
        import sqlite3

        con = sqlite3.connect(self.filename)
        cur = con.cursor()
        sqlite3_execute(
            cur, f"DELETE FROM RESULTS WHERE TIMESTAMP = '{self.timestamp}'", echo=echo
        )

        metadata = {
            "timestamp": self.name,
            "name": self.name,
            "category": self.category,
            "lr": self.learning_rate,
            "loss": self.loss,
        }

        QUERY_H = "INSERT INTO RESULTS( "
        QUERY_M = ") VALUES ("
        QUERY_T = ");"

        for key, value in metadata.items():
            QUERY_H += f"{key.upper()}, "
            if isinstance(value, float):
                QUERY_M += f"{value}, "
            else:
                QUERY_M += f"'{value}', "

        for key, value in self.data.items():
            QUERY_H += f"{key.upper()}, "
            if isinstance(value, float):
                QUERY_M += f"{value}, "
            else:
                v = str(value).replace("\n", "").replace("tensor", "").replace(" ", "")
                QUERY_M += f"'{v}', "

        QUERY_H = QUERY_H[:-2] + " "
        QUERY_M = QUERY_M[:-2] + " "
        QUERY = QUERY_H + QUERY_M + QUERY_T
        sqlite3_execute(cur, QUERY, echo=echo)
        con.commit()
        con.close()

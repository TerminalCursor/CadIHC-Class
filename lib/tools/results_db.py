from .call_graph import cg
from ..interfaces.db import Database
from typing import List, Self, Any
from sqlite3 import Cursor


def sqlite3_execute(cursor: Cursor, query: str, echo=False, dry_run=False):
    """# Execute Sqlite3 Query"""
    cg.add("sqlite3_execute")
    if echo:
        print(query)
    if not dry_run:
        cursor.execute(query)


class ResultsDatabase(Database):
    """# Results Database Class"""

    data: dict = {}
    meta: dict = {}

    scalar_result_keys: List[str] = [
        "total_accuracy",
        "training_accuracy",
        "validation_accuracy",
        "total_accuracy_c",
        "training_accuracy_c",
        "validation_accuracy_c",
    ]
    matrix_result_keys: List[str] = ["total_raw", "training_raw", "validation_raw"]
    meta_keys: List[str] = ["timestamp", "name", "category", "learning_rate", "loss"]

    def __init__(self: Self, filename: str):
        """# Set Filename to Manage"""
        cg.add("ResultsDatabase.__init__")
        super().__init__(filename)

    def add(self: Self, **results_dict: Any) -> Self:
        """# Add Data"""
        cg.add("ResultsDatabase.add")
        for key in results_dict.keys():
            keyl = key.lower()
            if keyl in self.meta_keys:
                self.meta[keyl] = results_dict[key]
            if keyl in self.scalar_result_keys:
                self.data[keyl] = float(results_dict[key])
            elif keyl in self.matrix_result_keys:
                self.data[keyl] = (
                    str(results_dict[key]).replace("array", "").replace("tensor", "")
                )
        return self

    def write(self: Self, echo: bool = False) -> Self:
        """# Insert Data

        From `self.data`
        """
        cg.add("ResultsDatabase.write")
        import sqlite3

        con = sqlite3.connect(self.filename)
        cur = con.cursor()
        sqlite3_execute(
            cur, f"DELETE FROM RESULTS WHERE TIMESTAMP = '{self.meta.get('timestamp')}'", echo=echo
        )

        metadata = {
            "timestamp": self.meta.get("timestamp"),
            "name": self.meta.get("name"),
            "category": self.meta.get("category"),
            "lr": self.meta.get("learning_rate"),
            "loss": self.meta.get("loss"),
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
        return self

from ..interfaces.db import Database
from typing import List, Dict, Self, Any
from sqlite3 import Cursor


def sqlite3_execute(
    cursor: Cursor, query: str, echo: bool = False, dry_run: bool = False
):
    """# Execute Sqlite3 Query"""
    if echo:
        print(query)
    if not dry_run:
        cursor.execute(query)


class ResultsDatabase(Database):
    """# Results Database Class"""

    data: Dict[str, str | int | float] = {}
    meta: Dict[str, str | int | float] = {}

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
        super().__init__(filename)

    def add(self: Self, **results_dict: Any) -> Self:
        """# Add Data"""
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
        import sqlite3

        con = sqlite3.connect(self.filename)
        cur = con.cursor()
        sqlite3_execute(
            cur,
            f"DELETE FROM RESULTS WHERE TIMESTAMP = '{self.meta.get('timestamp')}'",
            echo=echo,
        )

        metadata = {
            "timestamp": self.meta.get("timestamp"),
            "name": self.meta.get("name"),
            "category": self.meta.get("category"),
            "lr": self.meta.get("learning_rate"),
            "loss": self.meta.get("loss"),
        }

        query_H = "INSERT INTO RESULTS( "
        query_M = ") VALUES ("
        query_T = ");"

        for key, value in metadata.items():
            query_H += f"{key.upper()}, "
            if isinstance(value, float):
                query_M += f"{value}, "
            else:
                query_M += f"'{value}', "

        for key, value in self.data.items():
            query_H += f"{key.upper()}, "
            if isinstance(value, float):
                query_M += f"{value}, "
            else:
                v = str(value).replace("\n", "").replace("tensor", "").replace(" ", "")
                query_M += f"'{v}', "

        query_H = query_H[:-2] + " "
        query_M = query_M[:-2] + " "
        QUERY = query_H + query_M + query_T
        sqlite3_execute(cur, QUERY, echo=echo)
        con.commit()
        con.close()
        return self

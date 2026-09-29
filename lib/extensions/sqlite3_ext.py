# Sqlite3 Loading Extensions
from ..tools.call_graph import cg
from .pandas_ext import *
from ..interfaces.data.types import DataTypes
from typing import List, Tuple


def legacy_enumerate_model(
    series: Series | DataFrame | Any,
) -> Tuple[List[int], List[int]]:
    """# Convert Sql Column Values to Enumeration

    Returns ([enumeration of data type],
             [patch count if applicable else -1])
    """
    cg.add("legacy_enumerate_model")
    values = []
    evalues = []
    for value in series:
        data_type = DataTypes.Unknown
        extra_data = -1
        if value == "image":
            data_type |= DataTypes.ImageCNN
        if value[: len("image_vt")] == "image_vt":
            data_type |= DataTypes.ImageViT
            extra_data = int(value[len("image_vt") :])
        if "morphology" in value:
            data_type |= DataTypes.Morphology
        if "metrics" in value:
            data_type |= DataTypes.Morphology
        if "latent" in value:
            data_type |= DataTypes.Latent
        if "energetics" in value:
            data_type |= DataTypes.Energetics
        if value == "nuclei":
            data_type |= DataTypes.Morphology
            data_type |= DataTypes.Nuclei
        if value == "nuclei_only":
            data_type |= DataTypes.Nuclei
        if not data_type.is_valid():
            data_type = DataTypes.Unknown
        enumeration = data_type.value
        values.append(enumeration)
        evalues.append(extra_data)
    return values, evalues


def legacy_sql_load(
    sql_name: str, table_name: str, gen_columns: bool = True, fix_latent: bool = True
):
    """# Load SQLite3 Database to Pandas

    Additionally, generate some columns such as model enumeration
    """
    cg.add("legacy_sql_load")
    from pandas import read_sql_query
    from sqlite3 import connect

    conn = connect(sql_name)
    df = read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    if gen_columns:
        df["MODEL"], df["MODELe"] = legacy_enumerate_model(df["NAME"])
        df = df.time()
        if fix_latent:
            df = df[(df["TIME"] > 100_000) | (df["NAME"] != "latent")]
    df = df.reset_index(drop=True)
    return df

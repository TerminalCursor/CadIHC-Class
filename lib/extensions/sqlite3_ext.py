# Sqlite3 Loading Extensions
from .pandas_ext import *
from ..interfaces.data.types import DataTypes
from typing import List


def legacy_enumerate_model(series: Series | DataFrame | Any) -> List[int]:
    values = []
    for value in series:
        sub_values = []
        if value == "image":
            sub_values.append(DataTypes.ImageCNN)
        if value[: len("image_vt")] == "image_vt":
            sub_values.append(DataTypes.ImageViT)
            # sub_values.append(int(value[len('image_vt'):]))
        if "morphology" in value:
            sub_values.append(DataTypes.Morphology)
        if "metrics" in value:
            sub_values.append(DataTypes.Morphology)
        if "latent" in value:
            sub_values.append(DataTypes.Latent)
        if "energetics" in value:
            sub_values.append(DataTypes.Energetics)
        if value == "nuclei":
            sub_values.append(DataTypes.Morphology)
            sub_values.append(DataTypes.Nuclei)
        if value == "nuclei_only":
            sub_values.append(DataTypes.Nuclei)
        enumeration = sum([2 ** (i.value - 1) for i in sub_values])
        values.append(enumeration)
    return values


def legacy_sql_load(
    sql_name: str, table_name: str, gen_columns: bool = True, fix_latent: bool = True
):
    """# Load SQLite3 Database to Pandas

    Additionally, generate some columns such as model enumeration
    """
    from pandas import read_sql_query
    from sqlite3 import connect

    conn = connect(sql_name)
    df = read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    if gen_columns:
        # df["MODEL"] = df["NAME"].map(df["NAME"].enumerate())
        df["MODEL"] = legacy_enumerate_model(df["NAME"])
        df = df.time()
        if fix_latent:
            df = df[(df["TIME"] > 100_000) | (df["NAME"] != "latent")]
    df = df.reset_index(drop=True)
    return df

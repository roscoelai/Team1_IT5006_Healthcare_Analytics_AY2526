#!/usr/bin/env python
# etl.py
# 2025-08-27
# Roscoe

"""
Read raw data with the desired schema.

Write to Apache Parquet format. Moving forward, either read the parquet file,
or use the `read_raw()` function.
"""

import json
import os
import stat

import polars as pl


DF_SOURCE: str = "data/raw/diabetic_data.csv"
DD_SOURCE: str = "data/diabetes_datadict.csv"
IDS_MAPPING_SOURCE: str = "data/IDS_mapping.json"
DEST: str = "data/diabetic_data.parquet"


def read_raw(
    df_source: str=DF_SOURCE,
    dd_source: str=DD_SOURCE
) -> pl.DataFrame:
    """
    Use this function if reading from raw.
    """
    df = pl.read_csv(df_source, null_values=["?", "None"], infer_schema=False)
    dd = pl.read_csv(dd_source)

    exprs = []
    for row in dd.iter_rows(named=True):
        name = row["Variable"]
        if row["Dtype"] == "Enum":
            dct = json.loads(row["ValueCounts"])
            exprs.append(pl.col(name).cast(pl.Enum(dct.keys())))
        elif row["Dtype"] == "Int64":
            exprs.append(pl.col(name).cast(pl.Int64))
        elif row["Dtype"] != "String":
            pass
    df = df.with_columns(exprs)

    return df


def replace_ids(
    df: pl.DataFrame,
    ids_mapping_source: str=IDS_MAPPING_SOURCE
) -> pl.DataFrame:
    """
    To decide when is the best time to make the substitution, if at all.
    """
    with open(ids_mapping_source) as f:
        ids_mapping = json.load(f)
    for name, mapping in ids_mapping.items():
        df = df.with_columns(pl.col(name).replace(mapping))
    return df


def drop_constant_columns(df: pl.DataFrame) -> pl.DataFrame:
    const_cols = [s.name for s in df if s.n_unique() <= 1]
    print(f"Dropping constant columns: {const_cols}")
    return df.drop(const_cols)


def make_parquet(
    df: pl.DataFrame,
    dest: str=DEST,
    readonly: bool=True,
    verbose: bool=False
) -> None:
    if os.path.isfile(dest):
        os.chmod(dest, stat.S_IWRITE)
    df.write_parquet(dest)
    if readonly:
        os.chmod(dest, stat.S_IREAD)
    if verbose:
        msg = f"File written: '{dest}'"
        if readonly:
            msg += ", read-only"
        print(msg)



def main() -> None:
    df = read_raw()
    df = replace_ids(df)
    df = drop_constant_columns(df)
    make_parquet(df, verbose=True)


if __name__ == "__main__":
    main()



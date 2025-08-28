#!/usr/bin/env python
# etl.py
# 2025-08-28
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

import datadict


DF_SOURCE: str = "data/raw/diabetic_data.csv"
DD_SOURCE: str = "data/diabetes_datadict.csv"
DEST: str = "data/diabetic_data.parquet"


def read_raw(df_src: str=DF_SOURCE, dd_src: str=DD_SOURCE) -> pl.DataFrame:
    df = pl.read_csv(df_src, infer_schema=False)
    if not os.path.isfile(dd_src):
        datadict.make_and_write_datadict(df_src, dd_src, verbose=True)
    dd = pl.read_csv(dd_src)

    exprs = []
    for row in dd.iter_rows(named=True):
        name = row["Variable"]
        dtype = row["Dtype"]
        if dtype == "Enum":
            dct = json.loads(row["ValueCounts"])
            exprs.append(pl.col(name).cast(pl.Enum(dct.keys())))
        elif dtype == "Int64":
            exprs.append(pl.col(name).cast(int))
        elif dtype == "String":
            pass
        # elif dtype == "Categorical":
        #     exprs.append(pl.col(name).cast(pl.Categorical))
        else:
            print(f"WARNING: Unexpected dtype '{dtype}' for '{name}'.")
    df = df.with_columns(exprs)

    return df


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
    df = datadict.replace_ids(df)
    make_parquet(df, verbose=True)


if __name__ == "__main__":
    main()



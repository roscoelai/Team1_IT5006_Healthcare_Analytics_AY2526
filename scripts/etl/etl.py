#!/usr/bin/env python
# etl.py
# 2025-09-02
# Roscoe

"""
Read raw data with the desired schema.

Write to Apache Parquet format. Moving forward, either read the parquet file,
or use the `read_raw()` function.

TODO:
- [x] Data dictionary column names have changed, make corrections here
- [ ] Recode/Remap ID numbers
"""

import json
import os
import stat
import sys
from inspect import currentframe, getframeinfo

import polars as pl

sys.path.append(os.path.dirname(getframeinfo(currentframe()).filename))

import datadict


DF_SOURCE: str = "data/raw/diabetic_data.csv"
DD_SOURCE: str = "data/diabetes_datadict.csv"
DEST: str = "data/diabetic_data.parquet"


def read_raw(df_src: str=DF_SOURCE, dd_src: str=DD_SOURCE) -> pl.DataFrame:
    """
    Data dictionary is necessary to read from CSV properly. Optional for
    parquet since it can remember the schema.

    Use (keep as) pl.String instead of pl.Categorical for reduced file size.
    """
    df = pl.read_csv(df_src, infer_schema=False)
    if not os.path.isfile(dd_src):
        datadict.make_and_write_datadict(df_src, dd_src, verbose=True)
    dd = pl.read_csv(dd_src)
    exprs = []
    for row in dd.iter_rows(named=True):
        name = row["variable"]
        dtype = row["data_type"]
        if dtype == "Enum":
            dct = json.loads(row["value_counts"])
            exprs.append(pl.col(name).cast(pl.Enum(dct.keys())))
        elif dtype == "Int64":
            exprs.append(pl.col(name).cast(int))
        elif dtype == "String":
            pass
        else:
            print(f"WARNING: Unexpected dtype '{dtype}' for '{name}'.")
    df = df.with_columns(exprs)
    return df


def read_recoded(df_src: str=DF_SOURCE, dd_src: str=DD_SOURCE) -> pl.DataFrame:
    """
    - Read from CSV
    - Set proper data types
    - Recode ID columns
    - Redode ICD-9 columns
    """
    df = read_raw(df_src, dd_src)
    df = datadict.replace_ids(df)
    # TODO: `diag_1`, `diag_2`, and `diag_3`. Create new or replace?
    # NOTE: This is _NOT_ a one-to-one mapping, so we will lose information.
    df = datadict.replace_icd9s(df)
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
    df = read_recoded()
    make_parquet(df, verbose=True)


if __name__ == "__main__":
    main()



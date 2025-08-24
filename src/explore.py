#!/usr/bin/env python
# explore.py
# 2025-08-24
# Roscoe

import os

import polars as pl
import polars.selectors as cs


SOURCE: str = "data/diabetic_data.parquet"


def read_data(source: str=SOURCE) -> pl.DataFrame:
    if os.path.isfile(source):
        df = pl.read_parquet(source)
    else:
        from etl import etl
        df = etl.read_raw().pipe(etl.drop_constant_columns)
    return df



def main() -> None:
    df = read_data()
    print(df)


if __name__ == "__main__":
    main()



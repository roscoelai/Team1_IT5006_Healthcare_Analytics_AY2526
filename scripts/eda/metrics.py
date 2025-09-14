#!/usr/bin/env python
# metrics.py
# 2025-09-14
# Roscoe

import polars as pl


def calc_gini_impurity(s: pl.Series) -> float:
    vc = s.value_counts(normalize=True)
    vc = vc.with_columns(pl.col("proportion").pow(2).alias("prop2"))
    return 1 - vc["prop2"].sum()


def calc_entropy(s: pl.Series) -> float:
    p = pl.col("proportion")
    vc = s.value_counts(normalize=True)
    vc = vc.with_columns(-p.log(base=2).mul(p).alias("shannon"))
    return vc["shannon"].sum()



def main() -> None:
    pass


if __name__ == "__main__":
    main()



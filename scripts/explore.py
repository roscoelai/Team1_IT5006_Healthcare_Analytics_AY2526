#!/usr/bin/env python
# explore.py
# 2025-08-28
# Roscoe

"""
Have at it.
"""

import inspect
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
import seaborn as sns
from scipy.stats import chi2_contingency, spearmanr

sys.path.append(
    os.path.dirname(inspect.getframeinfo(inspect.currentframe()).filename)
)

from etl import datadict


SOURCE: str = "data/diabetic_data.parquet"


def read_data(source: str=SOURCE) -> pl.DataFrame:
    if os.path.isfile(source):
        df = pl.read_parquet(source)
    else:
        from etl import etl
        df = (etl.read_raw()
              .pipe(etl.replace_ids)
              .pipe(etl.drop_constant_columns))
        etl.make_parquet(df, verbose=True)
    return df


def calc_n_admissions(df: pl.DataFrame) -> pl.DataFrame:
    """
    Count number of encounters for each `patient_nbr`.
    Run separately if splitting.
    """
    vc = df["patient_nbr"].value_counts().rename({"count": "n_admissions"})
    df = df.drop("n_admissions", strict=False)
    df = df.join(vc, on="patient_nbr", how="left")
    lcols = ["encounter_id", "patient_nbr", "n_admissions"]
    df = df.select(*lcols, pl.exclude(lcols))
    return df


def calc_spearmanr(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """
    Spearman rank-order correlation coefficient.
    """
    arr = df.to_numpy()
    corr = spearmanr(arr)
    r = pl.DataFrame(corr.statistic).fill_nan(None)
    p = pl.DataFrame(corr.pvalue).fill_nan(None)
    r.columns = df.columns
    p.columns = df.columns
    if True:
        cols = [s.name for s in r if s.is_not_null().any()]
        rows = pl.any_horizontal(pl.all().is_not_null())
        r = r.select(cols).filter(rows)
        p = p.select(cols).filter(rows)
    return {"statistic": r, "pvalue": p}


def mask(df: pl.DataFrame, threshold: float) -> pl.DataFrame:
    """
    Change values below an absolute threshold to null.
    """
    for s in df:
        df = df.with_columns(pl.when(pl.col(s.name).abs().lt(threshold))
                             .then(pl.lit(None))
                             .otherwise(s.name)
                             .name.keep())
    return df


def viz_heatmap(
    df: pl.DataFrame,
    title: str,
    figpath: str | None=None,
    figsize: tuple[float, float]=(14, 12),
    annot: bool=True,
    fmt: str=".2g"
) -> plt.Axes:
    df = df.to_pandas()
    df.set_index(df.columns, inplace=True)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df, cmap="viridis", annot=True, fmt=fmt, ax=ax)
    ax.set_title(title)
    if figpath is not None:
        plt.savefig(figpath, dpi=300, bbox_inches="tight")
    plt.close()

# ---

def corr_numerics(df: pl.DataFrame, threshold: float=0.2) -> None:
    corr = df.select(cs.numeric(), pl.col("readmitted").cast(int)).corr()
    viz_heatmap(corr,
                title="Numeric variable correlations (Pearson, WARNING: `readmitted` treated as integers)",
                figpath="figs/pearson-numeric-only.png")
    corr2 = mask(corr, threshold=threshold)
    with pl.Config(tbl_cols=9) as cfg:
        print(corr2)


def corr_enums(df: pl.DataFrame, threshold: float=0.1) -> None:
    """
    Possible negative correlation between `age` and `A1Cresult`.
    The rest don't look so interesting at the moment.
    Wait, variable contains "None", which might be another problem...
    """
    enum_vals = df.select(cs.enum().to_physical())
    scorr = calc_spearmanr(enum_vals)["statistic"]
    viz_heatmap(scorr,
                title="Ordinal variable correlations (Spearman)",
                figpath="figs/spearman-ordinal-only.png",
                figsize=(16, 12),
                fmt=".2f")
    scorr2 = mask(scorr, threshold=threshold)
    with pl.Config(tbl_cols=26, tbl_rows=26) as cfg:
        print(scorr2)


def cramers_corrected_stat(confusion_matrix) -> float:
    """
    Calculate Cramers V statistic for categorial-categorial association.
    Uses correction from Bergsma and Wicher,
    Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = chi2_contingency(confusion_matrix).statistic
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))


def cramers_v_pairwise(df: pl.DataFrame, threshold: float=0.2) -> None:
    noms = df.select(cs.string(), "readmitted")
    noms = noms.select(pl.exclude("encounter_id", "patient_nbr"))
    df2 = noms

    names = noms.columns
    arr = np.empty((len(names), len(names)))
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names[i:], start=i):
            if n1 == n2:
                n2 = f"{n1}_copy"
                df3 = df2.select(n1, pl.col(n1).alias(n2))
            else:
                df3 = df2.select(n1, n2)
            counts = df3.group_by(n1, n2).len()
            counts = counts.pivot(on=n2, index=n1).fill_null(0)
            mat = counts.drop(n1).to_numpy()
            cramersv = cramers_corrected_stat(mat)
            arr[i, j] = cramersv
            arr[j, i] = cramersv
    df2 = pl.DataFrame(arr)
    df2.columns = names
    viz_heatmap(df2,
                title="Nominal variable Cramer's V",
                figpath="figs/cramersv-nominal-only.png")
    df3 = mask(df2, threshold=threshold)
    with pl.Config(tbl_cols=12, tbl_rows=12) as cfg:
        print(df3)



def main() -> None:
    df = read_data()
    df = calc_n_admissions(df)
    # print(df)
    with pl.Config(tbl_width_chars=300, fmt_str_lengths=200, tbl_rows=19) as cfg:
        for k in [f"diag_{i}" for i in [1, 2, 3]]:
            print(k)
            print(df[k].map_elements(datadict.icd9_lookup, return_dtype=str).value_counts(sort=True))
    return
    corr_numerics(df)
    corr_enums(df)
    cramers_v_pairwise(df)


if __name__ == "__main__":
    main()



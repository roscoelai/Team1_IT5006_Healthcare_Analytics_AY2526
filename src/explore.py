#!/usr/bin/env python
# explore.py
# 2025-08-26
# Roscoe

import os

import numpy as np
import polars as pl
import polars.selectors as cs
from scipy.stats import chi2_contingency, spearmanr


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

    If there is to be splitting, this should be run separately after splitting.
    """
    vc = df["patient_nbr"].value_counts().rename({"count": "n_admissions"})
    df = df.drop("n_admissions", strict=False)
    df = df.join(vc, on="patient_nbr", how="left")
    lcols = ["encounter_id", "patient_nbr", "n_admissions"]
    df = df.select(*lcols, pl.exclude(lcols))
    return df


def corr_numerics(df: pl.DataFrame, threshold: float=0.2) -> None:
    corr = df.select(cs.numeric()).corr()
    for s in corr:
        corr = corr.with_columns(pl.when(pl.col(s.name).abs().lt(threshold))
                                 .then(pl.lit(None))
                                 .otherwise(s.name)
                                 .name.keep())
    with pl.Config(tbl_cols=9) as cfg:
        print(corr)


def corr_enums(df: pl.DataFrame, threshold: float=0.1) -> None:
    """
    Possible negative correlation between `age` and `A1Cresult`.
    The rest don't look so interesting at the moment.
    Wait, variable contains "None", which might be another problem...
    """
    enums = df.select(cs.enum().to_physical())
    arr = enums.to_numpy()
    scorr = spearmanr(arr)
    # scorr = pl.DataFrame(scorr.pvalue).fill_nan(None)
    scorr = pl.DataFrame(scorr.statistic).fill_nan(None)
    scorr.columns = enums.columns
    scorr = scorr.select(s.name for s in scorr if not s.is_null().all())
    scorr = scorr.filter(pl.any_horizontal(pl.all().is_not_null()))
    for s in scorr:
        # scorr = scorr.with_columns(pl.when(pl.col(s.name).abs().ge(threshold))
        scorr = scorr.with_columns(pl.when(pl.col(s.name).abs().lt(threshold))
                                   .then(pl.lit(None))
                                   .otherwise(s.name)
                                   .name.keep())
    with pl.Config(tbl_cols=26, tbl_rows=26) as cfg:
        print(scorr)


def cramers_corrected_stat(confusion_matrix) -> float:
    """
    Calculate Cramers V statistic for categorial-categorial association.
    Uses correction from Bergsma and Wicher,
    Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    # chi2 = chi2_contingency(confusion_matrix)[0]
    chi2 = chi2_contingency(confusion_matrix).statistic
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))


def cramers_v_pairwise(df: pl.DataFrame, threshold: float=0.2) -> None:
    noms = df.select(cs.string())
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
    for s in df2:
        df2 = df2.with_columns(pl.when(pl.col(s.name).abs().lt(threshold))
                               .then(pl.lit(None))
                               .otherwise(s.name)
                               .name.keep())
    with pl.Config(tbl_cols=12, tbl_rows=12) as cfg:
        print(df2)
    return df2




def main() -> None:
    df = read_data()
    df = calc_n_admissions(df)
    # print(df)
    # corr_numerics(df)
    # corr_enums(df)
    cramers_v_pairwise(df)


if __name__ == "__main__":
    main()



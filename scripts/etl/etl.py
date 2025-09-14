#!/usr/bin/env python
# etl.py
# 2025-09-14
# Roscoe

"""
CSV files do not preserve schema. Reading directly from ZIP is slow. Set a
schema and write to Apache Parquet format. We will tolerate this intermediate
file for the duration of this project.

As part of the schema, set the dtypes for ordinal variables to `pl.Enum`,
which will mandate specifying the order of categories. Choose orders that make
sense. Unfortunately, there are missing values ("?") for `weight`. We place
them at the head of the order so it's obvious that there are missing values,
but know that this variable is unlikely to be used in any case as there are
too many missing values. We do something similar for `max_glu_serum` and
`A1Cresult`, mostly for the purpose of visualization.

Also, replace ID numbers with their descriptions. This will make EDA a lot
less painful, and also prevent these nominal variables from being accidentally
treated as numeric.
"""

import os
import stat
import sys
from inspect import currentframe, getframeinfo

import polars as pl

# Just in case. Assume `read_raw.py` is in the same directory.
_ = os.path.dirname(getframeinfo(currentframe()).filename)
if _ not in sys.path:
    sys.path.append(_)

from read_raw import read_data, get_maps

# -----------------------------------------------------------------------------

DEST: str = "data/diabetic_data.parquet"

_ENUM_MEDS: pl.Enum = pl.Enum(["No", "Down", "Steady", "Up"])

# Let's not micro-optimize dtypes, it's a tiny dataset.
SCHEMA: dict[str, type | pl.Enum] = {
    "encounter_id": int,  # Might imply order, so int instead of str.
    "patient_nbr": int,  # Just to be consistent with `encounter_id`.
    "race": str,
    "gender": str,
    "age": pl.Enum([f"[{i}-{i + 10})" for i in range(0, 100, 10)]),
    "weight": pl.Enum(["?"] +
                      [f"[{i}-{i + 25})" for i in range(0, 200, 25)] +
                      [">200"]),
    "admission_type_id": str,
    "discharge_disposition_id": str,
    "admission_source_id": str,
    "time_in_hospital": int,
    "payer_code": str,
    "medical_specialty": str,
    "num_lab_procedures": int,
    "num_procedures": int,
    "num_medications": int,
    "number_outpatient": int,
    "number_emergency": int,
    "number_inpatient": int,
    "diag_1": str,
    "diag_2": str,
    "diag_3": str,
    "number_diagnoses": int,
    "max_glu_serum": pl.Enum(["None", "Norm", ">200", ">300"]),
    "A1Cresult": pl.Enum(["None", "Norm", ">7", ">8"]),
    "metformin": _ENUM_MEDS,
    "repaglinide": _ENUM_MEDS,
    "nateglinide": _ENUM_MEDS,
    "chlorpropamide": _ENUM_MEDS,
    "glimepiride": _ENUM_MEDS,
    "acetohexamide": _ENUM_MEDS,
    "glipizide": _ENUM_MEDS,
    "glyburide": _ENUM_MEDS,
    "tolbutamide": _ENUM_MEDS,
    "pioglitazone": _ENUM_MEDS,
    "rosiglitazone": _ENUM_MEDS,
    "acarbose": _ENUM_MEDS,
    "miglitol": _ENUM_MEDS,
    "troglitazone": _ENUM_MEDS,
    "tolazamide": _ENUM_MEDS,
    "examide": _ENUM_MEDS,
    "citoglipton": _ENUM_MEDS,
    "insulin": _ENUM_MEDS,
    "glyburide-metformin": _ENUM_MEDS,
    "glipizide-metformin": _ENUM_MEDS,
    "glimepiride-pioglitazone": _ENUM_MEDS,
    "metformin-rosiglitazone": _ENUM_MEDS,
    "metformin-pioglitazone": _ENUM_MEDS,
    "change": str,
    "diabetesMed": str,
    "readmitted": pl.Enum(["NO", ">30", "<30"]),
}

# -----------------------------------------------------------------------------

def set_schema(
    df: pl.DataFrame,
    schema: dict[str, type | pl.Enum]=SCHEMA
) -> pl.DataFrame:
    return df.with_columns(pl.col(k).cast(v) for k, v in schema.items())


def remap_ids(
    df: pl.DataFrame,
    mappings: dict[str, dict[str, str]]=get_maps()
) -> pl.DataFrame:
    exprs = (pl.col(k).replace_strict(v) for k, v in mappings.items())
    return df.with_columns(exprs)


def remap_icd9(x: str) -> str:
    """
    Map specific ICD-9 codes into broader categories.

    https://en.wikipedia.org/wiki/List_of_ICD-9_codes
    """
    if x == "?":
        return x
    elif x.startswith("E"):
        # "External causes of injury"
        return "External Injury"
    elif x.startswith("V"):
        # "Supplemental classification"
        return "Supplemental classification"

    x_num = float(x)
    if 1 <= x_num <= 139:
        # "infectious and parasitic diseases"
        return "Infections"
    elif 140 <= x_num <= 239:
        # "neoplasms"
        return "Neoplasms"

    # -------------------------------------------------------------------------
    # Drill down into this category because diabetes is the focus.
    # "endocrine, nutritional and metabolic diseases, and immunity disorders"
    elif 249 <= x_num < 251:
        return "Diabetes mellitus"
    elif 240 <= x_num < 280:
        return "Other Endocrine/Metabolic/Immunity"
    # -------------------------------------------------------------------------

    elif 280 <= x_num <= 289:
        # "diseases of the blood and blood-forming organs"
        return "Blood"
    elif 290 <= x_num <= 319:
        # "mental disorders"
        return "Mental"
    elif 320 <= x_num <= 389:
        # "diseases of the nervous system and sense organs"
        return "Nervous"
    elif 390 <= x_num <= 459:
        # "diseases of the circulatory system"
        return "Circulatory"
    elif 460 <= x_num <= 519:
        # "diseases of the respiratory system"
        return "Respiratory"
    elif 520 <= x_num <= 579:
        # "diseases of the digestive system"
        return "Digestive"
    elif 580 <= x_num <= 629:
        # "diseases of the genitourinary system"
        return "Genitourinary"
    elif 630 <= x_num <= 679:
        # "complications of pregnancy, childbirth, and the puerperium"
        return "Pregnancy/Childbirth"
    elif 680 <= x_num <= 709:
        # "diseases of the skin and subcutaneous tissue"
        return "Dermatology"
    elif 710 <= x_num <= 739:
        # "diseases of the musculoskeletal system and connective tissue"
        return "Musculoskeletal"
    elif 740 <= x_num <= 759:
        # "congenital anomalies"
        return "Congenital"
    elif 760 <= x_num <= 779:
        # "certain conditions originating in the perinatal period"
        return "Perinatal"
    elif 780 <= x_num <= 799:
        # "symptoms, signs, and ill-defined conditions"
        return "Symptoms/Signs/Ill-defined"
    elif 800 <= x_num <= 999:
        # "injury and poisoning"
        return "Injury/Poisoning"
    return x


def remap_icd9s(df: pl.DataFrame) -> pl.DataFrame:
    exprs = (
        pl.col(k).map_elements(remap_icd9, return_dtype=str).alias(f"{k}b")
        for k in (f"diag_{i}" for i in range(1, 4))
    )
    return df.with_columns(exprs)


def preprocess_raw(rows: list[dict[str, str]]=read_data()) -> pl.DataFrame:
    df = (pl.DataFrame(rows)
        .pipe(set_schema)
        .pipe(remap_ids)
        .pipe(remap_icd9s)
    )
    return df


def make_parquet(df: pl.DataFrame, dest: str=DEST) -> None:
    if os.path.isfile(dest):
        os.chmod(dest, stat.S_IWRITE)
    df.write_parquet(dest)
    os.chmod(dest, stat.S_IREAD)
    print(f"File written: '{dest}'")



def main() -> None:
    df = preprocess_raw()
    make_parquet(df)


if __name__ == "__main__":
    main()



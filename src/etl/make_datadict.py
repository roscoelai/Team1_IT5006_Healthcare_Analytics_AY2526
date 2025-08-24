#!/usr/bin/env python
# make_datadict.py
# 2025-08-24
# Roscoe

"""
Create a data dictionary for the dataset.

Columns:
- Categories taken from the project brief
- Descriptions taken from the dataset website
- Data types will be declared here and recorded
- Summary counts for the number of non-blank values per column
- Number of unique values per column
- Value counts for each column
  - Stringified dictionaries if the number of categories is small
"""

import json
import os
import stat

import polars as pl
import polars.selectors as cs


SOURCE: str = "data/raw/diabetic_data.csv"
IDS_MAPPING_SOURCE: str = "data/IDS_mapping.json"
DESCRIPTIONS_SOURCE: str = "data/descriptions.json"
DEST: str = "data/diabetes_datadict.csv"


def define_categories() -> dict[str, str]:
    """
    Group into categories. This might be useful.
    """
    categories = {
        "encounter_id": "Identifiers",
        "patient_nbr": "Identifiers",
        "race": "Demographics",
        "gender": "Demographics",
        "age": "Demographics",
        "weight": "Demographics",
        "admission_type_id": "Admission Details",
        "discharge_disposition_id": "Admission Details",
        "admission_source_id": "Admission Details",
        "time_in_hospital": "Admission Details",
        "payer_code": "Healthcare Provider",
        "medical_specialty": "Healthcare Provider",
        "num_lab_procedures": "Clinical Metrics",
        "num_procedures": "Clinical Metrics",
        "num_medications": "Clinical Metrics",
        "number_outpatient": "Clinical Metrics",
        "number_emergency": "Clinical Metrics",
        "number_inpatient": "Clinical Metrics",
        "diag_1": "Diagnoses",
        "diag_2": "Diagnoses",
        "diag_3": "Diagnoses",
        "number_diagnoses": "Clinical Metrics",
        "max_glu_serum": "Laboratory Results",
        "A1Cresult": "Laboratory Results",
        "metformin": "Medications",
        "repaglinide": "Medications",
        "nateglinide": "Medications",
        "chlorpropamide": "Medications",
        "glimepiride": "Medications",
        "acetohexamide": "Medications",
        "glipizide": "Medications",
        "glyburide": "Medications",
        "tolbutamide": "Medications",
        "pioglitazone": "Medications",
        "rosiglitazone": "Medications",
        "acarbose": "Medications",
        "miglitol": "Medications",
        "troglitazone": "Medications",
        "tolazamide": "Medications",
        "examide": "Medications",
        "citoglipton": "Medications",
        "insulin": "Medications",
        "glyburide-metformin": "Medications",
        "glipizide-metformin": "Medications",
        "glimepiride-pioglitazone": "Medications",
        "metformin-rosiglitazone": "Medications",
        "metformin-pioglitazone": "Medications",
        "change": "Treatment Changes",
        "diabetesMed": "Treatment Changes",
        "readmitted": "Target Variables",
    }
    return categories


def define_schema(df: pl.DataFrame) -> pl.DataFrame:
    """
    The default data type will be strings, to be treated as normative measures.
    Specify which variables should be considered numeric and which should be
    considered ordinal.
    """
    categories = define_categories()

    # Numerics.
    integers = [k for k, v in categories.items()
                if k == "time_in_hospital" or v == "Clinical Metrics"]

    # Enums (ordered categoricals).
    medication_categories = ["No", "Down", "Steady", "Up"]
    enums = {
        "age": ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
                "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"],
        # "weight": ["[0-25)", "[25-50)", "[50-75)", "[75-100)", "[100-125)",
        #            "[125-150)", "[150-175)", "[175-200)", ">200", "?"],
        "weight": ["[0-25)", "[25-50)", "[50-75)", "[75-100)", "[100-125)",
                   "[125-150)", "[150-175)", "[175-200)", ">200"],
        "max_glu_serum": ["None", "Norm", ">200", ">300"],
        "A1Cresult": ["None", "Norm", ">7", ">8"],
        "metformin": medication_categories,
        "repaglinide": medication_categories,
        "nateglinide": medication_categories,
        "chlorpropamide": medication_categories,
        "glimepiride": medication_categories,
        "acetohexamide": medication_categories,
        "glipizide": medication_categories,
        "glyburide": medication_categories,
        "tolbutamide": medication_categories,
        "pioglitazone": medication_categories,
        "rosiglitazone": medication_categories,
        "acarbose": medication_categories,
        "miglitol": medication_categories,
        "troglitazone": medication_categories,
        "tolazamide": medication_categories,
        "examide": medication_categories,
        "citoglipton": medication_categories,
        "insulin": medication_categories,
        "glyburide-metformin": medication_categories,
        "glipizide-metformin": medication_categories,
        "glimepiride-pioglitazone": medication_categories,
        "metformin-rosiglitazone": medication_categories,
        "metformin-pioglitazone": medication_categories,
        "readmitted": ["NO", ">30", "<30"],
    }

    df = df.with_columns(pl.col(integers).cast(int))
    df = df.with_columns(pl.col(k).cast(pl.Enum(v)) for k, v in enums.items())
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


def make_datadict(
    df: pl.DataFrame,
    descriptions_source: str=DESCRIPTIONS_SOURCE
) -> pl.DataFrame:
    with open(descriptions_source) as f:
        descs = json.load(f)
    descs = {k: descs[k] for k in df.columns}
    categories = define_categories()
    dtypes = []
    summary_counts = []
    n_uniques = []
    vcds = []
    for s in df:
        dtypes.append(str(type(s.dtype)))
        summary_counts.append(s.is_not_null().sum())
        n_uniques.append(s.n_unique())
        vc = s.value_counts().sort(s.name)
        if vc.height <= 118:
            vcd = dict(zip(vc[s.name], vc["count"]))
            if s.dtype == pl.Enum:
                vcd = {k: vcd.get(k, 0) for k in s.dtype.categories}
            vcds.append(str(vcd).replace("'", '"'))
        else:
            vcds.append(None)
    pdd = pl.DataFrame({
        "Variable": df.columns,
        "Category": categories.values(),
        "Description": descs.values(),
        "Dtype": dtypes,
        "SummaryCount": summary_counts,
        "NUnique": n_uniques,
        "ValueCounts": vcds,
    })
    return pdd



def make_and_write_datadict(
    source: str=SOURCE,
    dest: str=DEST,
    ids_mapping_source: str=IDS_MAPPING_SOURCE,
    descriptions_source: str=DESCRIPTIONS_SOURCE,
    readonly: bool=True,
    verbose: bool=False
) -> None:
    df = (pl.read_csv(SOURCE, null_values="?", infer_schema=False)
          .pipe(define_schema)
          .pipe(replace_ids, ids_mapping_source=ids_mapping_source))
    dd = make_datadict(df, descriptions_source=descriptions_source)
    if os.path.isfile(dest):
        os.chmod(dest, stat.S_IWRITE)
    dd.write_csv(dest)
    if readonly:
        os.chmod(dest, stat.S_IREAD)
    if verbose:
        msg = f"File written: '{dest}'"
        if readonly:
            msg += ", read-only"
        print(msg)



def main() -> None:
    make_and_write_datadict(verbose=True)


if __name__ == "__main__":
    main()



#!/usr/bin/env python
# datadict.py
# 2025-09-14
# Roscoe

"""
Create a data dictionary for the dataset.

Variable categories from the project brief. Descriptions from UCI ML
Repository, used to be hard-coded here, but moved out to `descriptions.json`
for simpler maintenance. It adds a dependency, so be careful. We will decide
the schema.

This doesn't really flow nicely because the data dictionary will involve
some summaries of the data, and that has to happen after preprocessing.

The data dictionary should minimally provide the descriptions for each
variable. Additionally, it could also specify data types and variable
categories. Summaries can include counting unique values per variable,
counting number of missing values per variable, even perhaps a breakdown of
counts of each unique value if space permits.

Extra bells and whistles could include calculating the proportion of mode
values per variable, entropy, or Gini impurity index, though they might
encroach on the domain of EDA, so the inclusion of these are up for debate.
"""

import json
import os
import stat
import sys
from enum import Enum
from inspect import currentframe, getframeinfo

import polars as pl

# This will be necessary to import modules from a different folder.
_ = os.path.dirname(getframeinfo(currentframe()).filename)
if _ not in sys.path:
    sys.path.append(_)
_ = os.path.dirname(_)
if _ not in sys.path:
    sys.path.append(_)

from eda.metrics import calc_entropy, calc_gini_impurity
from etl import preprocess_raw


DESCRIPTIONS_JSON: str = "data/descriptions.json"
DEST: str = "data/diabetes_datadict.csv"


class FeatureType(str, Enum):
    IDENTIFIER = "Identifier"
    TEXT = "Text"
    CONTINUOUS = "Continuous"
    DISCRETE = "Discrete"
    NOMINAL = "Nominal"
    ORDINAL = "Ordinal"
    DATETIME = "Datetime"
    BOOLEAN = "Boolean"


class Categories(str, Enum):
    IDENTIFIERS = "Identifiers"
    DEMOGRAPHICS = "Demographics"
    ADMISSION_DETAILS = "Admission Details"
    HEALTHCARE_PROVIDER = "Healthcare Provider"
    CLINICAL_METRICS = "Clinical Metrics"
    DIAGNOSES = "Diagnoses"
    LABORATORY_RESULTS = "Laboratory Results"
    MEDICATIONS = "Medications"
    TREATMENT_CHANGES = "Treatment Changes"
    TARGET_VARIABLES = "Target Variables"


MISSING_VALUES = {
    "encounter_id": False,
    "patient_nbr": False,
    "race": True,
    "gender": False,
    "age": False,
    "weight": True,
    "admission_type_id": True,
    "discharge_disposition_id": True,
    "admission_source_id": True,
    "time_in_hospital": False,
    "payer_code": True,
    "medical_specialty": True,
    "num_lab_procedures": False,
    "num_procedures": False,
    "num_medications": False,
    "number_outpatient": False,
    "number_emergency": False,
    "number_inpatient": False,
    "diag_1": True,
    "diag_2": True,
    "diag_3": True,
    "number_diagnoses": False,
    "max_glu_serum": False,
    "A1Cresult": False,
    "metformin": False,
    "repaglinide": False,
    "nateglinide": False,
    "chlorpropamide": False,
    "glimepiride": False,
    "acetohexamide": False,
    "glipizide": False,
    "glyburide": False,
    "tolbutamide": False,
    "pioglitazone": False,
    "rosiglitazone": False,
    "acarbose": False,
    "miglitol": False,
    "troglitazone": False,
    "tolazamide": False,
    "examide": False,
    "citoglipton": False,
    "insulin": False,
    "glyburide-metformin": False,
    "glipizide-metformin": False,
    "glimepiride-pioglitazone": False,
    "metformin-rosiglitazone": False,
    "metformin-pioglitazone": False,
    "change": False,
    "diabetesMed": False,
    "readmitted": False,
    "diag_1b": True,
    "diag_2b": True,
    "diag_3b": True,
}


FEATURE_TYPES = {
    "encounter_id": FeatureType.IDENTIFIER,
    "patient_nbr": FeatureType.IDENTIFIER,
    "race": FeatureType.NOMINAL,
    "gender": FeatureType.NOMINAL,
    "age": FeatureType.ORDINAL,
    "weight": FeatureType.ORDINAL,
    "admission_type_id": FeatureType.NOMINAL,
    "discharge_disposition_id": FeatureType.NOMINAL,
    "admission_source_id": FeatureType.NOMINAL,
    "time_in_hospital": FeatureType.DISCRETE,
    "payer_code": FeatureType.NOMINAL,
    "medical_specialty": FeatureType.NOMINAL,
    "num_lab_procedures": FeatureType.DISCRETE,
    "num_procedures": FeatureType.DISCRETE,
    "num_medications": FeatureType.DISCRETE,
    "number_outpatient": FeatureType.DISCRETE,
    "number_emergency": FeatureType.DISCRETE,
    "number_inpatient": FeatureType.DISCRETE,
    "diag_1": FeatureType.NOMINAL,
    "diag_2": FeatureType.NOMINAL,
    "diag_3": FeatureType.NOMINAL,
    "number_diagnoses": FeatureType.DISCRETE,
    "max_glu_serum": FeatureType.NOMINAL,  # Ordinal if not for "None"
    "A1Cresult": FeatureType.NOMINAL,  # Ordinal if not for "None"
    "metformin": FeatureType.ORDINAL,
    "repaglinide": FeatureType.ORDINAL,
    "nateglinide": FeatureType.ORDINAL,
    "chlorpropamide": FeatureType.ORDINAL,
    "glimepiride": FeatureType.ORDINAL,
    "acetohexamide": FeatureType.ORDINAL,
    "glipizide": FeatureType.ORDINAL,
    "glyburide": FeatureType.ORDINAL,
    "tolbutamide": FeatureType.ORDINAL,
    "pioglitazone": FeatureType.ORDINAL,
    "rosiglitazone": FeatureType.ORDINAL,
    "acarbose": FeatureType.ORDINAL,
    "miglitol": FeatureType.ORDINAL,
    "troglitazone": FeatureType.ORDINAL,
    "tolazamide": FeatureType.ORDINAL,
    "examide": FeatureType.ORDINAL,
    "citoglipton": FeatureType.ORDINAL,
    "insulin": FeatureType.ORDINAL,
    "glyburide-metformin": FeatureType.ORDINAL,
    "glipizide-metformin": FeatureType.ORDINAL,
    "glimepiride-pioglitazone": FeatureType.ORDINAL,
    "metformin-rosiglitazone": FeatureType.ORDINAL,
    "metformin-pioglitazone": FeatureType.ORDINAL,
    "change": FeatureType.BOOLEAN,
    "diabetesMed": FeatureType.BOOLEAN,
    "readmitted": FeatureType.ORDINAL,
    "diag_1b": FeatureType.NOMINAL,
    "diag_2b": FeatureType.NOMINAL,
    "diag_3b": FeatureType.NOMINAL,
}

VARIABLE_CATEGORIES: dict[str, str] = {
    "encounter_id": Categories.IDENTIFIERS,
    "patient_nbr": Categories.IDENTIFIERS,
    "race": Categories.DEMOGRAPHICS,
    "gender": Categories.DEMOGRAPHICS,
    "age": Categories.DEMOGRAPHICS,
    "weight": Categories.DEMOGRAPHICS,
    "admission_type_id": Categories.ADMISSION_DETAILS,
    "discharge_disposition_id": Categories.ADMISSION_DETAILS,
    "admission_source_id": Categories.ADMISSION_DETAILS,
    "time_in_hospital": Categories.ADMISSION_DETAILS,
    "payer_code": Categories.HEALTHCARE_PROVIDER,
    "medical_specialty": Categories.HEALTHCARE_PROVIDER,
    "num_lab_procedures": Categories.CLINICAL_METRICS,
    "num_procedures": Categories.CLINICAL_METRICS,
    "num_medications": Categories.CLINICAL_METRICS,
    "number_outpatient": Categories.CLINICAL_METRICS,
    "number_emergency": Categories.CLINICAL_METRICS,
    "number_inpatient": Categories.CLINICAL_METRICS,
    "diag_1": Categories.DIAGNOSES,
    "diag_2": Categories.DIAGNOSES,
    "diag_3": Categories.DIAGNOSES,
    "number_diagnoses": Categories.CLINICAL_METRICS,
    "max_glu_serum": Categories.LABORATORY_RESULTS,
    "A1Cresult": Categories.LABORATORY_RESULTS,
    "metformin": Categories.MEDICATIONS,
    "repaglinide": Categories.MEDICATIONS,
    "nateglinide": Categories.MEDICATIONS,
    "chlorpropamide": Categories.MEDICATIONS,
    "glimepiride": Categories.MEDICATIONS,
    "acetohexamide": Categories.MEDICATIONS,
    "glipizide": Categories.MEDICATIONS,
    "glyburide": Categories.MEDICATIONS,
    "tolbutamide": Categories.MEDICATIONS,
    "pioglitazone": Categories.MEDICATIONS,
    "rosiglitazone": Categories.MEDICATIONS,
    "acarbose": Categories.MEDICATIONS,
    "miglitol": Categories.MEDICATIONS,
    "troglitazone": Categories.MEDICATIONS,
    "tolazamide": Categories.MEDICATIONS,
    "examide": Categories.MEDICATIONS,
    "citoglipton": Categories.MEDICATIONS,
    "insulin": Categories.MEDICATIONS,
    "glyburide-metformin": Categories.MEDICATIONS,
    "glipizide-metformin": Categories.MEDICATIONS,
    "glimepiride-pioglitazone": Categories.MEDICATIONS,
    "metformin-rosiglitazone": Categories.MEDICATIONS,
    "metformin-pioglitazone": Categories.MEDICATIONS,
    "change": Categories.TREATMENT_CHANGES,
    "diabetesMed": Categories.TREATMENT_CHANGES,
    "readmitted": Categories.TARGET_VARIABLES,
    "diag_1b": Categories.DIAGNOSES,
    "diag_2b": Categories.DIAGNOSES,
    "diag_3b": Categories.DIAGNOSES,
}


def get_descs(source: str=DESCRIPTIONS_JSON) -> dict[str, str]:
    with open(source) as f:
        return json.load(f)


def make_skeleton(descriptions: dict[str, str]=get_descs()) -> pl.DataFrame:
    dd = pl.DataFrame(
        {
            "variable": descriptions.keys(),
            "category": [VARIABLE_CATEGORIES[k] for k in descriptions],
            "description": descriptions.values(),
            "feature_type": [FEATURE_TYPES[k] for k in descriptions],
        }
    )
    return dd


def make_datadict(df: pl.DataFrame | None = None) -> pl.DataFrame:
    dd = make_skeleton()
    if df is None:
        return dd
    missing_set = {"?", "Not Available", "Not Mapped", "NULL"}
    dtypes = []
    n_uniques = []
    vcds = []
    n_missings = []
    for s in df:
        dtypes.append(str(type(s.dtype)))
        n_uniques.append(s.n_unique())
        vc = s.value_counts().sort(s.name)
        n_missing = s.is_null().sum()
        if s.dtype in [pl.String, pl.Enum]:
            ms = missing_set
            if s.name in {"discharge_disposition_id", "admission_type_id"}:
                ms = missing_set | {"Unknown/Invalid"}
            n_missing += s.cast(str).is_in(ms).sum()

        n_missings.append(n_missing)
        if vc.height <= 118:
            vcd = dict(zip(vc[s.name], vc["count"]))
            if s.dtype == pl.Enum:
                vcd = {k: vcd.get(k, 0) for k in s.dtype.categories}
            vcds.append(str(vcd).replace("'", '"'))  # For valid JSON format.
        else:
            vcds.append(None)
    dd2 = pl.DataFrame(
        {
            "variable": df.columns,
            "data_type": dtypes,
            "value_counts": vcds,
            "n_unique": n_uniques,
            "missing_values": n_missings,
            "mode_pct": [
                s.value_counts(normalize=True)["proportion"].max() for s in df
            ],
            "gini_impurity": [round(calc_gini_impurity(s), 12) for s in df],
            "shannon_entropy": [round(calc_entropy(s), 12) for s in df],
        }
    )
    dd = dd.join(dd2, on="variable", how="left")
    return dd


def make_and_write_datadict(dest: str = DEST) -> None:
    df = preprocess_raw()
    dd = make_datadict(df)
    if os.path.isfile(dest):
        os.chmod(dest, stat.S_IWRITE)
    dd.write_csv(dest)
    os.chmod(dest, stat.S_IREAD)
    print(f"File written: '{dest}'")


def main() -> None:
    make_and_write_datadict()


if __name__ == "__main__":
    main()

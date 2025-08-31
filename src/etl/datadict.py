#!/usr/bin/env python
# datadict.py
# 2025-08-31
# Roscoe

"""
Create a data dictionary for the dataset.
"""

import json
import os
import stat

import polars as pl


SOURCE: str = "data/raw/diabetic_data.csv"
DEST: str = "data/diabetes_datadict.csv"

# ---

VARIABLE_CATEGORIES: dict[str, str] = {
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

_RX_DESC: str = ("The feature indicates whether the drug was prescribed or "
                 "there was a change in the dosage. Values: up if the dosage "
                 "was increased during the encounter, down if the dosage was "
                 "decreased, steady if the dosage did not change, and no if "
                 "the drug was not prescribed")

DESCRIPTIONS: dict[str, str] = {
    "encounter_id": "Unique identifier of an encounter",
    "patient_nbr": "Unique identifier of a patient",
    "race": "Values: Caucasian, Asian, African American, Hispanic, and other",
    "gender": "Values: male, female, and unknown/invalid",
    "age": "Grouped in 10-year intervals: [0, 10), [10, 20),..., [90, 100)",
    "weight": "Weight in pounds.",
    "admission_type_id": ("Integer identifier corresponding to 9 distinct "
                          "values, for example, emergency, urgent, elective, "
                          "newborn, and not available"),
    "discharge_disposition_id": ("Integer identifier corresponding to 29 "
                                 "distinct values, for example, discharged "
                                 "to home, expired, and not available"),
    "admission_source_id": ("Integer identifier corresponding to 21 distinct "
                            "values, for example, physician referral, "
                            "emergency room, and transfer from a hospital"),
    "time_in_hospital": ("Integer number of days between admission and "
                         "discharge"),
    "payer_code": ("Integer identifier corresponding to 23 distinct values, "
                   "for example, Blue Cross/Blue Shield, Medicare, and "
                   "self-pay"),
    "medical_specialty": ("Integer identifier of a specialty of the admitting "
                          "physician, corresponding to 84 distinct values, "
                          "for example, cardiology, internal medicine, "
                          "family/general practice, and surgeon"),
    "num_lab_procedures": "Number of lab tests performed during the encounter",
    "num_procedures": ("Number of procedures (other than lab tests) performed "
                       "during the encounter"),
    "num_medications": ("Number of distinct generic names administered during "
                        "the encounter"),
    "number_outpatient": ("Number of outpatient visits of the patient in the "
                          "year preceding the encounter"),
    "number_emergency": ("Number of emergency visits of the patient in the "
                         "year preceding the encounter"),
    "number_inpatient": ("Number of inpatient visits of the patient in the "
                         "year preceding the encounter"),
    "diag_1": ("The primary diagnosis (coded as first three digits of ICD9); "
               "848 distinct values"),
    "diag_2": ("Secondary diagnosis (coded as first three digits of ICD9); "
               "923 distinct values"),
    "diag_3": ("Additional secondary diagnosis (coded as first three digits "
               "of ICD9); 954 distinct values"),
    "number_diagnoses": "Number of diagnoses entered to the system",
    "max_glu_serum": ("Indicates the range of the result or if the test was "
                      "not taken. Values: >200, >300, normal, and none if not "
                      "measured"),
    "A1Cresult": ("Indicates the range of the result or if the test was not "
                  "taken. Values: >8 if the result was greater than 8%, >7 if "
                  "the result was greater than 7% but less than 8%, normal if "
                  "the result was less than 7%, and none if not measured."),
    "metformin": _RX_DESC,
    "repaglinide": _RX_DESC,
    "nateglinide": _RX_DESC,
    "chlorpropamide": _RX_DESC,
    "glimepiride": _RX_DESC,
    "acetohexamide": _RX_DESC,
    "glipizide": _RX_DESC,
    "glyburide": _RX_DESC,
    "tolbutamide": _RX_DESC,
    "pioglitazone": _RX_DESC,
    "rosiglitazone": _RX_DESC,
    "acarbose": _RX_DESC,
    "miglitol": _RX_DESC,
    "troglitazone": _RX_DESC,
    "tolazamide": _RX_DESC,
    "examide": _RX_DESC,
    "citoglipton": _RX_DESC,
    "insulin": _RX_DESC,
    "glyburide-metformin": _RX_DESC,
    "glipizide-metformin": _RX_DESC,
    "glimepiride-pioglitazone": _RX_DESC,
    "metformin-rosiglitazone": _RX_DESC,
    "metformin-pioglitazone": _RX_DESC,
    "change": ("Indicates if there was a change in diabetic medications "
               "(either dosage or generic name). Values: change and no "
               "change"),
    "diabetesMed": ("Indicates if there was any diabetic medication "
                    "prescribed. Values: yes and no"),
    "readmitted": ("Days to inpatient readmission. Values: <30 if the patient "
                   "was readmitted in less than 30 days, >30 if the patient "
                   "was readmitted in more than 30 days, and No for no record "
                   "of readmission."),
}

IDS_MAPPINGS: dict[str, dict[str, str]] = {
    "admission_type_id": {
        "1": "Emergency",
        "2": "Urgent",
        "3": "Elective",
        "4": "Newborn",
        "5": "Not Available",
        "6": "NULL",
        "7": "Trauma Center",
        "8": "Not Mapped",
    },
    "discharge_disposition_id": {
        "1": "Discharged to home",
        "2": "Discharged/transferred to another short term hospital",
        "3": "Discharged/transferred to SNF",
        "4": "Discharged/transferred to ICF",
        "5": ("Discharged/transferred to another type of inpatient care "
              "institution"),
        "6": "Discharged/transferred to home with home health service",
        "7": "Left AMA",
        "8": "Discharged/transferred to home under care of Home IV provider",
        "9": "Admitted as an inpatient to this hospital",
        "10": "Neonate discharged to another hospital for neonatal aftercare",
        "11": "Expired",
        "12": "Still patient or expected to return for outpatient services",
        "13": "Hospice / home",
        "14": "Hospice / medical facility",
        "15": ("Discharged/transferred within this institution to Medicare "
               "approved swing bed"),
        "16": ("Discharged/transferred/referred another institution for "
               "outpatient services"),
        "17": ("Discharged/transferred/referred to this institution for "
               "outpatient services"),
        "18": "NULL",
        "19": "Expired at home. Medicaid only, hospice.",
        "20": "Expired in a medical facility. Medicaid only, hospice.",
        "21": "Expired, place unknown. Medicaid only, hospice.",
        "22": ("Discharged/transferred to another rehab fac including rehab "
               "units of a hospital."),
        "23": "Discharged/transferred to a long term care hospital.",
        "24": ("Discharged/transferred to a nursing facility certified under "
               "Medicaid but not certified under Medicare."),
        "25": "Not Mapped",
        "26": "Unknown/Invalid",
        "30": ("Discharged/transferred to another Type of Health Care "
               "Institution not Defined Elsewhere"),
        "27": "Discharged/transferred to a federal health care facility.",
        "28": ("Discharged/transferred/referred to a psychiatric hospital of "
               "psychiatric distinct part unit of a hospital"),
        "29": "Discharged/transferred to a Critical Access Hospital (CAH).",
    },
    "admission_source_id": {
        "1": "Physician Referral",
        "2": "Clinic Referral",
        "3": "HMO Referral",
        "4": "Transfer from a hospital",
        "5": "Transfer from a Skilled Nursing Facility (SNF)",
        "6": "Transfer from another health care facility",
        "7": "Emergency Room",
        "8": "Court/Law Enforcement",
        "9": "Not Available",
        "10": "Transfer from critial access hospital",
        "11": "Normal Delivery",
        "12": "Premature Delivery",
        "13": "Sick Baby",
        "14": "Extramural Birth",
        "15": "Not Available",
        "17": "NULL",
        "18": "Transfer From Another Home Health Agency",
        "19": "Readmission to Same Home Health Agency",
        "20": "Not Mapped",
        "21": "Unknown/Invalid",
        "22": "Transfer from hospital inpt/same fac reslt in a sep claim",
        "23": "Born inside this hospital",
        "24": "Born outside this hospital",
        "25": "Transfer from Ambulatory Surgery Center",
        "26": "Transfer from Hospice",
    }
}


def icd9_lookup(x: str) -> str:
    """
    https://en.wikipedia.org/wiki/List_of_ICD-9_codes

    Will need to look more carefully at this one...
    """
    # These may not be interesting.
    if x == "?":
        return x
    elif x.startswith("V"):
        return "supplemental classification"
    elif x.startswith("E"):
        return "external causes of injury "

    x_num = float(x)
    if 1 <= x_num <= 139:
        return "infectious and parasitic diseases"
    elif 140 <= x_num <= 239:
        return "neoplasms"

    # Drill down into this category because diabetes is the focus.
    # elif 240 <= x_num <= 279:
    #     return ("endocrine, nutritional and metabolic diseases, and immunity "
    #             "disorders")
    elif 240 <= x_num < 247:
        return "Disorders of thyroid gland"
    elif 249 <= x_num < 250:
        return "Secondary diabetes mellitus"
    elif 250 <= x_num < 251:
        return "Diabetes mellitus"
    elif 251 <= x_num < 260:
        return "Diseases of other endocrine glands"
    # elif 249 <= x_num < 260:
    #     return "Diseases of other endocrine glands"
    elif 260 <= x_num < 270:
        return "Nutritional deficiencies"
    elif 270 <= x_num < 280:
        return "Other metabolic and immunity disorders"

    elif 280 <= x_num <= 289:
        return "diseases of the blood and blood-forming organs"
    elif 290 <= x_num <= 319:
        return "mental disorders"
    elif 320 <= x_num <= 389:
        return "diseases of the nervous system and sense organs"
    elif 390 <= x_num <= 459:
        return "diseases of the circulatory system"
    elif 460 <= x_num <= 519:
        return "diseases of the respiratory system"
    elif 520 <= x_num <= 579:
        return "diseases of the digestive system"
    elif 580 <= x_num <= 629:
        return "diseases of the genitourinary system"
    elif 630 <= x_num <= 679:
        return "complications of pregnancy, childbirth, and the puerperium"
    elif 680 <= x_num <= 709:
        return "diseases of the skin and subcutaneous tissue"
    elif 710 <= x_num <= 739:
        return "diseases of the musculoskeletal system and connective tissue"
    elif 740 <= x_num <= 759:
        return "congenital anomalies"
    elif 760 <= x_num <= 779:
        return "certain conditions originating in the perinatal period"
    elif 780 <= x_num <= 799:
        return "symptoms, signs, and ill-defined conditions"
    elif 800 <= x_num <= 999:
        return "injury and poisoning"
    return x


def make_skeleton() -> pl.DataFrame:
    dd = pl.DataFrame({
        "Variable": DESCRIPTIONS.keys(),
        "Category": [VARIABLE_CATEGORIES[k] for k in DESCRIPTIONS],
        "Description": DESCRIPTIONS.values(),
    })
    return dd


def set_schema(df: pl.DataFrame) -> pl.DataFrame:
    """
    Define and set data types for variables.

    Questionable to treat "?" or "None" as first ordinals, but will mainly be
    for visualization. Unlikely these variables will be used as-is.
    """
    numerics = [
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
    ]

    medications_ord = ["No", "Down", "Steady", "Up"]
    enums = {
        "age": [
            "[0-10)",
            "[10-20)",
            "[20-30)",
            "[30-40)",
            "[40-50)",
            "[50-60)",
            "[60-70)",
            "[70-80)",
            "[80-90)",
            "[90-100)",
        ],
        "weight": [
            "?",
            "[0-25)",
            "[25-50)",
            "[50-75)",
            "[75-100)",
            "[100-125)",
            "[125-150)",
            "[150-175)",
            "[175-200)",
            ">200",
        ],
        "max_glu_serum": ["None", "Norm", ">200", ">300"],
        "A1Cresult": ["None", "Norm", ">7", ">8"],
        "metformin": medications_ord,
        "repaglinide": medications_ord,
        "nateglinide": medications_ord,
        "chlorpropamide": medications_ord,
        "glimepiride": medications_ord,
        "acetohexamide": medications_ord,
        "glipizide": medications_ord,
        "glyburide": medications_ord,
        "tolbutamide": medications_ord,
        "pioglitazone": medications_ord,
        "rosiglitazone": medications_ord,
        "acarbose": medications_ord,
        "miglitol": medications_ord,
        "troglitazone": medications_ord,
        "tolazamide": medications_ord,
        "examide": medications_ord,
        "citoglipton": medications_ord,
        "insulin": medications_ord,
        "glyburide-metformin": medications_ord,
        "glipizide-metformin": medications_ord,
        "glimepiride-pioglitazone": medications_ord,
        "metformin-rosiglitazone": medications_ord,
        "metformin-pioglitazone": medications_ord,
        "readmitted": ["NO", ">30", "<30"],
    }

    df = df.with_columns(pl.col(numerics).cast(int))
    df = df.with_columns(pl.col(k).cast(pl.Enum(v)) for k, v in enums.items())
    return df


def replace_ids(df: pl.DataFrame) -> pl.DataFrame:
    """
    To decide when is the best time to make the substitution.
    """
    for name, mapping in IDS_MAPPINGS.items():
        df = df.with_columns(pl.col(name).replace(mapping))
    return df


def calc_gini_impurity(s: pl.Series) -> float:
    vc = s.value_counts(normalize=True)
    vc = vc.with_columns(pl.col("proportion").pow(2).alias("prop2"))
    return 1 - vc["prop2"].sum()


def calc_entropy(s: pl.Series) -> float:
    """
    KIV.
    """
    p = pl.col("proportion")
    vc = s.value_counts(normalize=True)
    vc = vc.with_columns(-p.log(base=2).mul(p).alias("shannon"))
    return vc["shannon"].sum()


def make_datadict(df: pl.DataFrame | None=None) -> pl.DataFrame:
    dd = make_skeleton()
    if df is None:
        return dd
    dtypes = []
    n_uniques = []
    ginis = []
    modepcts = []
    vcds = []
    for s in df:
        dtypes.append(str(type(s.dtype)))
        n_uniques.append(s.n_unique())
        ginis.append(calc_gini_impurity(s))
        vc = s.value_counts().sort(s.name)
        modepcts.append(vc["count"].max() / df.height)
        if vc.height <= 118:
            vcd = dict(zip(vc[s.name], vc["count"]))
            if s.dtype == pl.Enum:
                vcd = {k: vcd.get(k, 0) for k in s.dtype.categories}
            vcds.append(str(vcd).replace("'", '"'))
        else:
            vcds.append(None)
    dd2 = pl.DataFrame({
        "Variable": df.columns,
        "Dtype": dtypes,
        "NUnique": n_uniques,
        "GiniImpurity": ginis,
        "ModePct": modepcts,
        "ValueCounts": vcds,
    })
    dd = dd.join(dd2, on="Variable", how="left")
    return dd


def make_and_write_datadict(
    source: str=SOURCE,
    dest: str=DEST,
    readonly: bool=True,
    verbose: bool=False
) -> None:
    df = (pl.read_csv(SOURCE, infer_schema=False)
          .pipe(replace_ids)
          .pipe(set_schema))
    dd = make_datadict(df)
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



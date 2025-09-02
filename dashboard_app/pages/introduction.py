import os
import json
from functools import partial

import pandas as pd
import streamlit as st

st.title("Introduction")

# ------------------------------------------------------------------------------
# 1. Objective
# ------------------------------------------------------------------------------
st.subheader("Objective")
st.markdown(
    """
    The key objective of this project is to analyze the **"Diabetes 130-US Hospitals 
    (1999-2008)"** dataset to predict the **30-day readmissions risk (target variable)**
    """
)

# ------------------------------------------------------------------------------
# 2. Dataset Overview
# ------------------------------------------------------------------------------
st.subheader("Dataset Overview")
st.markdown(
    """
    - **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)  
    - **Domain**:  Healthcare/Hospital Management  
    - **Size**: 101,766 records of hospitalized patients diagnosed with diabetes across 130 hospitals  
    - **Time Period**: 1999 - 2008  
    - **Recommended Train-test split**: None  
    - **Sensitive Data**: Age, gender and race of patients  
    - **Inclusion Criteria**:  
        - It is an inpatient encounter (a hospital admission)  
        - It is a diabetic encounter, that is, one during which any kind of diabetes was entered into the system as a diagnosis  
        - The length of stay was at least 1 day and at most 14 days  
        - Laboratory tests were performed during the encounter  
        - Medications were administered during the encounter  

    """
)


@st.cache_data
def load_descr_df(path: str) -> pd.DataFrame:
    with open(path) as f:
        descriptions = json.load(f)
    return pd.DataFrame(list(descriptions.items()), columns=["Feature", "Description"])


def style_df(df: pd.DataFrame, target_feature: str) -> pd.DataFrame:

    def highlight_target(row: pd.Series, target_feature: str) -> list[str]:
        return [
            "background-color: blue" if row["Feature"] == target_feature else ""
            for _ in row
        ]

    styled_df = df.style.apply(
        partial(highlight_target, target_feature=target_feature), axis=1
    )
    return styled_df


current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "../data")

descriptions_df = load_descr_df(os.path.join(data_dir, "descriptions.json"))
styled_df = style_df(descriptions_df, target_feature="readmitted")
st.dataframe(styled_df, width="content")
# ------------------------------------------------------------------------------
# 3. Feature Categories
# ------------------------------------------------------------------------------
st.subheader("Feature Categories")
st.markdown(
    """
    - Identifiers
    - IDs are usually not of interest, but the patients might not be unique in this dataset
    - Demographics
    - `race` and `weight` have missing values, `weight` might have to be dropped
    - Admission Details
    - Healthcare Provider
    - `payer_code` and `medical_specialty` have missing values
    - Clinical Metrics
    - Diagnoses
    - `diag_1`, `diag_2`, and `diag_3` have missing values
    - Laboratory Results
    - Medications
    - Treatment Changes
    - Target Variables
    """
)

# ------------------------------------------------------------------------------
# 4. Data Types
# ------------------------------------------------------------------------------
st.subheader("Data Types")
st.markdown(
    """
    - All data types are either integer or categorical
    - `encounter_id`, `patient_nbr`, `admission_type_id`, `discharge_disposition_id`, and `admission_source_id` are integers that should not be treated as numbers.
    - Ordinal: `age`, `max_glu_serum`, `A1Cresult`, Medications, and possibly `readmitted` (but might need to binarize, unless doing multinomial classification?)

    """
)

# ------------------------------------------------------------------------------
# 5. Footer
# ------------------------------------------------------------------------------
st.markdown(
    """
    The [Diabetes 130-US Hospitals (1999-2008)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) dataset[^1][^2] is maintained by the [UC Irvine (UCI) Machine Learning Repository](https://archive.ics.uci.edu/). It is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license. Which allows for the sharing and adaptation of the datasets for any purpose, provided that the appropriate credit is given.
    [^1]: Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). Diabetes 130-US Hospitals for Years 1999-2008 [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5230J.

    [^2]: Beata Strack, Jonathan P. DeShazo, Chris Gennings, Juan L. Olmo, Sebastian Ventura, Krzysztof J. Cios, and John N. Clore, "Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records", BioMed Research International, vol. 2014, Article ID 781670, 11 pages, 2014.
    """
)

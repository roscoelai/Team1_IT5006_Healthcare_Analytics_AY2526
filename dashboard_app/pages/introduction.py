import os
import json
from functools import partial

import pandas as pd
import streamlit as st

from components.overview_card import OverviewCard
from components.datadict_card import DataDictCard
from utils.dataloader import DataLoader

# Load data required for the page
df = DataLoader.get_data()
metadata = DataLoader.get_metadata()


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
    The dataset is from the UCI Machine Learning Repository which contains `101,766` 
    hospital encounters from 130 U.S. hospitals, spanning 1999–2008, with `50` 
    variables covering demographics, diagnoses, procedures, labs, and medications.  

    The target variable is `readmitted`, which indicates whether a patient was 
    readmitted within 30 days, after 30 days, or not readmitted.  

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
OverviewCard(df, metadata).render()

# ------------------------------------------------------------------------------
# Data Dictionary
# ------------------------------------------------------------------------------
st.subheader("Data Dictionary")
st.markdown(
    """
    The data dictionary provides metadata about each variable in the dataset,
    including its name, type, category, and description.
    """
)
DataDictCard(metadata).render()

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

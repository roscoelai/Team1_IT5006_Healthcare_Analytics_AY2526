import os
import json
from functools import partial

import pandas as pd
import streamlit as st

from components.overview_card import OverviewCard


# TODO: move this to shared folder
@st.cache_data
def load_csv_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "../data")
df = load_csv_data(os.path.join(data_dir, "diabetic_data.csv"))
metadata = load_csv_data(os.path.join(data_dir, "diabetes_datadict.csv"))

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
st.info(
    """🔍 Use the filters on the left to explore specific feature types or categories. 
    Click on the column headers to sort by a specific variable."""
)

col1, col2 = st.columns([2, 5])

# TODO: too much duplicate code. refactor later.
st.session_state["feature_type_filter"] = metadata["feature_type"].unique().tolist()
st.session_state["category_filter"] = metadata["category"].unique().tolist()
with col1:
    with st.container(border=True, height="stretch"):

        st.subheader("**Filter**")

        # Filter by feature type
        with st.expander("By Feature Type"):
            with st.container(horizontal=True, horizontal_alignment="right"):
                select_all_feat = st.button(
                    label="Select All",
                    key="select_all_feat",
                    type="tertiary",
                    use_container_width=False,
                )

                clear_all_feat = st.button(
                    label="Clear All",
                    key="clear_all_feat",
                    type="tertiary",
                    use_container_width=False,
                )
                if select_all_feat:
                    st.session_state["feature_type_filter"] = (
                        metadata["feature_type"].unique().tolist()
                    )
                    st.rerun()
                if clear_all_feat:
                    st.session_state["feature_type_filter"] = []
                    st.rerun()

            feature_type_options = metadata["feature_type"].unique().tolist()
            feature_type_filter = []
            for option in feature_type_options:
                checked = st.checkbox(
                    option,
                    value=option in st.session_state.get("feature_type_filter", []),
                    key=f"feature_type_{option}",
                )
                if checked:
                    feature_type_filter.append(option)
            st.session_state["feature_type_filter"] = feature_type_filter

        # Filter by category
        with st.expander("By Feature Category"):
            with st.container(horizontal=True, horizontal_alignment="right"):
                select_all_cat = st.button(
                    label="Select All",
                    key="select_all_cat",
                    type="tertiary",
                    use_container_width=False,
                )
                clear_all_feature_types = st.button(
                    label="Clear All",
                    key="dclear_all_feature_types",
                    type="tertiary",
                    use_container_width=False,
                )
                if select_all_cat:
                    st.session_state["category_filter"] = (
                        metadata["category"].unique().tolist()
                    )
                    st.rerun()
                if clear_all_feature_types:
                    st.session_state["category_filter"] = []
                    st.rerun()

            category_options = metadata["category"].unique().tolist()
            category_filter = []
            for option in category_options:
                checked = st.checkbox(
                    option,
                    value=option in st.session_state.get("category_filter", []),
                    key=f"category_{option}",
                )
                if checked:
                    category_filter.append(option)
            st.session_state["category_filter"] = category_filter

with col2:
    datadict = metadata[
        metadata["feature_type"].isin(st.session_state.get("feature_type_filter", []))
        & metadata["category"].isin(st.session_state.get("category_filter", []))
    ]
    datadict.reset_index(inplace=True, drop=True)
    datadict.index += 1  # Start index from 1 instead of 0
    datadict = datadict.drop(columns=["value_counts", "data_type"])
    filtered = len(st.session_state.get("feature_type_filter", [])) != len(
        feature_type_options
    ) or len(st.session_state.get("category_filter", [])) != len(category_options)
    st.write(
        f'<i>Total variables: {datadict.shape[0]} {"(filtered)" if filtered else ""}</i>',
        unsafe_allow_html=True,
    )
    st.write(datadict)

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

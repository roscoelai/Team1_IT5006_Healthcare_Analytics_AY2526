import os
from enum import Enum

import pandas as pd
import streamlit as st
import plotly.express as px

from dashboard_app.components.categorical_card import CategoricalCard
from dashboard_app.components.numerical_card import NumericalCard
from dashboard_app.components.identifier_card import IdentifierCard
from dashboard_app.components.num_corr_card import NumericalCorrelationCard
from dashboard_app.components.cat_corr_card import CategoricalCorrelationCard
from dashboard_app.components.num_cat_corr_card import NumericalCategoricalCorrCard
from dashboard_app.components.overview_card import OverviewCard
from dashboard_app.constants.feature_type import FeatureType


# ---
# TODO: move this to shared folder
@st.cache_data
def load_csv_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "../data")
df = load_csv_data(os.path.join(data_dir, "diabetic_data.csv"))
metadata = load_csv_data(os.path.join(data_dir, "diabetes_datadict.csv"))

# ---

st.title("Data Preprocessing")

# ------------------------------------------------------------------------------
# 1. Raw Data Overview
# ------------------------------------------------------------------------------
st.header("📄 Overview of Raw Data")
st.markdown(
    """
    """
)
st.warning("TODO: include some written summary description of the raw data")
OverviewCard(df, metadata).render()

# ------------------------------------------------------------------------------
# Univariate Analysis
# ------------------------------------------------------------------------------
st.subheader("🔍 Univariate Analysis")
st.warning("TODO: add some written summary description of univariate analysis")

selected_var = st.selectbox(
    "Select a variable to view metrics:", metadata["variable"].tolist()
)
data_dict_series = metadata[metadata["variable"] == selected_var]
feature_type = data_dict_series["feature_type"].values[0]

match feature_type:
    case FeatureType.IDENTIFIER:
        IdentifierCard(df[selected_var], data_dict_series).render()
    case FeatureType.DISCRETE | FeatureType.CONTINUOUS:
        NumericalCard(df[selected_var], data_dict_series).render()
    case FeatureType.NOMINAL | FeatureType.ORDINAL:
        CategoricalCard(df[selected_var], data_dict_series).render()
    case FeatureType.BOOLEAN:
        CategoricalCard(df[selected_var], data_dict_series).render()
    case _:
        st.warning(f"Feature type '{feature_type}' is not supported.")


# ------------------------------------------------------------------------------
# Bivariate / Multivariate Analysis
# ------------------------------------------------------------------------------
st.subheader("🔍 Bivariate / Multivariate Analysis")

tab1, tab2, tab3 = st.tabs(
    [
        "Numerical <-> Numerical",
        "Categorical <-> Categorical",
        "Numerical <-> Categorical",
    ]
)

with tab1:
    NumericalCorrelationCard(df, metadata).render()
with tab2:
    CategoricalCorrelationCard(df, metadata).render()
with tab3:
    NumericalCategoricalCorrCard(df, metadata).render()

# ------------------------------------------------------------------------------
# 2. Overview of Data Preprocessing Steps
# ------------------------------------------------------------------------------
st.subheader("🧹 Overview of Data Preprocessing Steps")
st.markdown(
    """
    - <Include summary of data preprocessing steps here>  
    - <links to subsections below>  
    - <tldr stuff>  
    """
)


# ------------------------------------------------------------------------------
# 3. Treatment of Missing Values
# ------------------------------------------------------------------------------
st.subheader("❓ Treatment of Missing Values ")
has_missing = metadata[metadata["missing_values"] > 0]
has_missing["percentage_missing"] = has_missing["missing_values"] / df.shape[0] * 100
has_missing = has_missing.sort_values("percentage_missing", ascending=False)
has_missing["percentage_missing"] = has_missing["percentage_missing"].map(
    lambda x: f"{x:.2f}%"
)

# Plot missing values
fig = px.bar(
    has_missing,
    x="variable",
    y="percentage_missing",
    title="Variables with Missing Values",
    text=has_missing["percentage_missing"],
    labels={"percentage_missing": "Percentage Missing"},
)
st.plotly_chart(fig)

st.markdown(
    f"""
    There are a total of **seven** variables in the dataset with missing values.  

    - `weight` has a very high proportion of missing values 
    **({has_missing[has_missing['variable'] == 'weight']['percentage_missing'].values[0]})**. 
    The approach here is to **drop the column** entirely.
    - 
    """
)

has_missing[["strategy", "rationale"]] = ["label as 'Unknown'", ""]
has_missing.loc[has_missing["variable"] == "weight", ["strategy", "rationale"]] = [
    "drop",
    "High proportion of missing values",
]
has_missing.loc[has_missing["variable"] == "payer_code", ["strategy", "rationale"]] = [
    "drop",
    "Payment method should have no impact on readmission",
]

st.dataframe(
    # has_missing[["variable", "description", "percentage_missing", "strategy"]],
    has_missing[
        [
            "variable",
            "feature_type",
            "description",
            "percentage_missing",
            "strategy",
            "rationale",
        ]
    ],
    hide_index=True,
)

# ------------------------------------------------------------------------------
# 4. Encoding of Categorical variables
# ------------------------------------------------------------------------------
st.subheader("🔡 Encoding of Categorical variables")
st.markdown(
    """
    - one hot encoding?  
    - ordinal encoding?  
    - label encoding?  
    - frequency encoding?  
    - should probably be some model agnostic encoding for now until we know what model to use?  
    - others ...  
    """
)

# ------------------------------------------------------------------------------
# 5. Detection and Handling of Outliers
# ------------------------------------------------------------------------------
st.subheader("🚫 Detection and Handling of Outliers")
st.markdown(
    """
    - show off some outlier detection methods (iqr, z-score, box plot, histogram, scatterplot, etc.)  
    - outlier detection methods try to show off both visual and non-visual methods  
    - should show the before and after choice of preprocessing  
    - explain how we have decided to handl outliers  
        - remove the entire row?  
        - replace with nearest percentile?  
        - are the outliers legit or just some error?  
        - leave as it?  
        - JUSTIFY WHY  
    """
)

# ------------------------------------------------------------------------------
# 6. Feature Scaling
# ------------------------------------------------------------------------------
st.subheader("📏 Feature Scaling")
st.markdown(
    """
    - standardization / normalization?  
    """
)

# ------------------------------------------------------------------------------
# 7. Feature Engineering
# ------------------------------------------------------------------------------
st.subheader("🧠 Feature Engineering")
st.markdown(
    """
    - dimensionality reduction?  
    - PCA?  
    - encode target label  
    - drop the duplicates?  
    - low variance / high correlation filter?
    - feature selection  
    - probably not now, until we understand the data better first
    """
)

# ------------------------------------------------------------------------------
# 6. Data after Preprocessing
# ------------------------------------------------------------------------------
st.subheader("✅ Data after Preprocessing")
st.markdown(
    """
    - describe the final data after preprocessing
    - describe
    """
)

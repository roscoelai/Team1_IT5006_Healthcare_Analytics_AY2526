import os
from enum import Enum

import pandas as pd
import streamlit as st
import plotly.express as px

from dashboard_app.components.categorical_card import CategoricalCard
from dashboard_app.components.numerical_card import NumericalCard
from dashboard_app.components.identifier_card import IdentifierCard
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
st.header("Overview of Raw Data 📄")
st.markdown(
    """
    <TODO: include some written summary description of the raw data>
    """
)
OverviewCard(df, metadata).render()

# Use data dict to show all key metrics for a single variable
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
st.markdown(
    """
    - basically just mention how we handle missing values.    
        - drop the rows with missing values?  
        - remove entire column with too many missing values?  
        - mean/median/mode imputation?  
        - replace with a new category ("unknown") for categorical vals?  
        - something more advanced? k nearest neighbour? reg models? train model to impute?
        - need to justify strategy e.g. too many missing values? etc etc...  
        strategy (e.g. lit review suggest average is sth?), etc etc...  
    """
)

# plot barchart of missing values
missing_values = df.isnull().sum()
st.bar_chart(missing_values)

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

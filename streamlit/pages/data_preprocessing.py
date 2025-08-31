import streamlit as st

st.title("Data Preprocessing")

# ------------------------------------------------------------------------------
# 1. Raw Data Overview
# ------------------------------------------------------------------------------
st.subheader("📄 Raw Data Overview")
st.markdown(
    """
    - <Include summary statistics here>  
    - <Include visualisation on raw data here>

    - num rows & columns  
    - include sample rows  
    - summary statistics  
    - Make a note on placeholers for missing values in the raw data (e.g. "?")  
    - Visualization on count of categorical vs numerical data  
    - <suggest other visualisation / descriptions>  
    """
)


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

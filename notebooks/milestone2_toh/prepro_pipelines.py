"""Preprocessing pipelines for the diabetes readmission dataset.

This module includes preprocessing pipelines that are used to clean and prepare
the diabetes readmission dataset for modeling.

The preprocessing performed are agnostic to the model used, and can be reused across
different model pipelines.

All the pipelines return a pandas DataFrame, and can be chained together using the pandas
.pipe() method.

Example:
    df = (
        load_data("data/diabetes_data.csv")
        .pipe(keep_first_encounter)
        .pipe(drop_columns, ["weight", "payer_code"])
        .pipe(replace_missing, "max_glu_serum", "None")
        ...
    )
"""

import os
import pandas as pd


def load_data(input_path: str) -> pd.DataFrame:
    """Load data from a CSV file.

    Args:
        input_path (str): Path to the input CSV file.
    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame.
    Raises:
        Exception: If there is an error loading the data.
    """
    print(f"=== Loading data from {input_path} ===")
    try:
        data = pd.read_csv(input_path)
        print(f"Data loaded with shape: {data.shape}")
        return data
    except Exception as e:
        print(f"[ERROR] Error loading data: {e}")
        raise


def replace_missing(
    data: pd.DataFrame,
    column: str,
    replace_value: any,
) -> pd.DataFrame:
    """Replace missing values in a specified column with a given value.

    Args:
        data (pd.DataFrame): Input DataFrame.
        column (str): Column name to replace missing values in.
        replace_value (any): Value to replace missing values with.
    Returns:
        pd.DataFrame: DataFrame with missing values replaced.
    """
    print(f"=== Replacing missing values in column: {column} with {replace_value} ===")
    data.fillna(value=replace_value, inplace=True)
    return data


def drop_columns(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Drop specified columns from the DataFrame.

    Args:
        data (pd.DataFrame): Input DataFrame.
        columns (list): List of column names to drop.
    Returns:
        pd.DataFrame: DataFrame with specified columns dropped.
    """

    print("=== Dropping columns ===")

    # Drop specified columns if they exist in the DataFrame
    existing_columns = [col for col in columns if col in data.columns]

    # Drop the existing columns from the DataFrame
    data.drop(columns=existing_columns, inplace=True)

    print(
        f"Remaining columns:\n    • " + "\n    • ".join(data.columns),
    )
    return data


def remap_column(data: pd.DataFrame, column: str, mapping: dict) -> pd.DataFrame:
    """Remap values in a specified column based on a given mapping.

    Args:
        data (pd.DataFrame): Input DataFrame.
        column (str): Column name to remap values in.
        mapping (dict): Dictionary mapping old values to new values.
    Returns:
        pd.DataFrame: DataFrame with remapped values in the specified column.
    """
    print(f"=== Remapping values in column: {column} ===")
    if column in data.columns:
        data.replace({column: mapping}, inplace=True)
        print(f"Remapped values in column: {column}")
    return data


def drop_row_with_value(
    data: pd.DataFrame,
    column: str,
    value: any,
) -> pd.DataFrame:
    """Drop rows where a specified column has a given value.
    Args:
        data (pd.DataFrame): Input DataFrame.
        column (str): Column name to check for the value.
        value (any): Value to drop rows for.
    Returns:
        pd.DataFrame: DataFrame with rows dropped where the specified column has the given value.
    """
    print(f"=== Dropping rows with value '{value}' in column: {column} ===")

    # Drop rows where the specified column has the value
    initial_shape = data.shape
    data = data[data[column] != value].reset_index(drop=True)
    final_shape = data.shape
    print(
        f"Dropped {initial_shape[0] - final_shape[0]} rows with value '{value}' in column '{column}'",
    )
    return data


def keep_first_encounter(data: pd.DataFrame) -> pd.DataFrame:
    """Keep only the first encounter for each patient based on encounter_id.
    Args:
        data (pd.DataFrame): Input DataFrame.
    Returns:
        pd.DataFrame: DataFrame with only the first encounter for each patient.
    """
    print("=== Keeping only the first encounter for each patient ===")

    # Keep only the first encounter for each patient
    initial_shape = data.shape
    data = (
        data.sort_values(by=["patient_nbr", "encounter_id"])
        .drop_duplicates(subset=["patient_nbr"], keep="first")
        .reset_index(drop=True)
    )
    final_shape = data.shape
    print(
        f"Initial shape: {initial_shape}, after keeping first encounter: {final_shape}",
    )
    return data


def to_csv(data: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """Save DataFrame to a CSV file.

    Args:
        data (pd.DataFrame): DataFrame to save.
        output_path (str): Path to save the CSV file.
    Returns:
        pd.DataFrame: The same DataFrame that was saved.
    Raises:
        Exception: If there is an error saving the data.
    """
    print(f"=== Saving data to {output_path} ===")
    try:
        # Create the directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save the DataFrame to a CSV file
        data.to_csv(output_path, index=False)
        print(f"Data saved to {output_path} | shape: {data.shape}")
        return data
    except Exception as e:
        print(f"[ERROR] Error saving data: {e}")
        raise


def coarse_class_diagnosis(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """Coarse class diagnosis codes in a specified column.
    Args:
        data (pd.DataFrame): Input DataFrame.
        column (str): Column name containing diagnosis codes to coarse class.
    Returns:
        pd.DataFrame: DataFrame with coarsely classified diagnosis codes in the specified column.
    """
    print(f"=== Coarse classing diagnosis codes in column: {column} ===")

    def classify_icd9(code):
        try:
            if pd.isnull(code):
                return "Unknown"
            code_str = str(code)
            if code_str.startswith("E"):
                return "External causes of injury"
            if code_str.startswith("V"):
                return "Supplemental classification"
            code_float = float(code_str)
            if 1 <= code_float <= 139:
                return "Infectious and parasitic diseases"
            elif 140 <= code_float <= 239:
                return "Neoplasms"
            elif 240 <= code_float <= 279:
                return "Endocrine, nutritional and metabolic diseases, and immunity disorders"
            elif 280 <= code_float <= 289:
                return "Diseases of the blood and blood-forming organs"
            elif 290 <= code_float <= 319:
                return "Mental disorders"
            elif 320 <= code_float <= 389:
                return "Diseases of the nervous system and sense organs"
            elif 390 <= code_float <= 459:
                return "Diseases of the circulatory system"
            elif 460 <= code_float <= 519:
                return "Diseases of the respiratory system"
            elif 520 <= code_float <= 579:
                return "Diseases of the digestive system"
            elif 580 <= code_float <= 629:
                return "Diseases of the genitourinary system"
            elif 630 <= code_float <= 679:
                return "Complications of pregnancy, childbirth, and the puerperium"
            elif 680 <= code_float <= 709:
                return "Diseases of the skin and subcutaneous tissue"
            elif 710 <= code_float <= 739:
                return "Diseases of the musculoskeletal system and connective tissue"
            elif 740 <= code_float <= 759:
                return "Congenital anomalies"
            elif 760 <= code_float <= 779:
                return "Certain conditions originating in the perinatal period"
            elif 780 <= code_float <= 799:
                return "Symptoms, signs, and ill-defined conditions"
            elif 800 <= code_float <= 999:
                return "Injury and poisoning"
            else:
                return "Other"
        except Exception:
            return "Unknown"

    if column in data.columns:
        data[column] = data[column].apply(classify_icd9)
        print(f"Coarse classed diagnosis codes in column: {column}")
    return data

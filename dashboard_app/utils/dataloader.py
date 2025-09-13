import pandas as pd
import streamlit as st


@st.cache_data
def _load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# TODO: catch io exceptions - in this case file is assumed to be there
class DataLoader:
    _data_path: str = None
    _metadata_path: str = None

    @staticmethod
    def init(data_path: str, metadata_path: str):
        """Initialize the DataLoader with paths to data and metadata CSV files.

        Args:
            data_path (str): Path to the main dataset CSV file.
            metadata_path (str): Path to the metadata CSV file.
        """
        DataLoader._data_path = data_path
        DataLoader._metadata_path = metadata_path

    @staticmethod
    def get_data() -> pd.DataFrame | None:
        """Get the main dataset as a pandas DataFrame.

        Returns:
            pd.DataFrame | None: The main dataset if the path is set, otherwise None.
        """
        if DataLoader._data_path is None:
            return None
        return _load_csv(DataLoader._data_path)

    @staticmethod
    def get_metadata() -> pd.DataFrame | None:
        """Get the metadata as a pandas DataFrame.
        Returns:
            pd.DataFrame | None: The metadata if the path is set, otherwise None.
        """
        # Lazy load the metadata
        if DataLoader._metadata_path is None:
            return None
        return _load_csv(DataLoader._metadata_path)

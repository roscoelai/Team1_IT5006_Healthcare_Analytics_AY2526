from enum import Enum

import pandas as pd
import streamlit as st

from constants.feature_type import FeatureType


class DataSubset(str, Enum):
    ALL = "All Data"
    FIRST_100 = "First 100 Rows"
    RANDOM_SAMPLE = "Random Sample"


class OverviewCard:
    """A card that provides an overview of the dataset, including key metrics
    and a sample of the data."""

    def __init__(self, data: pd.DataFrame, metadata: pd.DataFrame):
        self.data = data
        self.metadata = metadata
        self._compute_summary()

    def render(self):

        # Data Overview
        with st.container(border=True):
            self._render_key_metrics()
            st.info(
                """🔍 Use the dropdown below to select the subset of data to view. """
            )
            option = st.selectbox(
                "Select subset of raw data to view:",
                options=[member.value for member in DataSubset],
                index=1,
            )
            col1, col2 = st.columns([1, 5])
            with col2:
                self._render_sample_data(option)
            with col1:
                self._render_data_overview()

    def _compute_summary(self):
        """Compute summary statistics required for the overview card."""
        self.num_nominal = self.metadata[
            self.metadata["feature_type"] == FeatureType.NOMINAL
        ].shape[0]
        self.num_ordinal = self.metadata[
            self.metadata["feature_type"] == FeatureType.ORDINAL
        ].shape[0]
        self.num_continuous = self.metadata[
            self.metadata["feature_type"] == FeatureType.CONTINUOUS
        ].shape[0]
        self.num_discrete = self.metadata[
            self.metadata["feature_type"] == FeatureType.DISCRETE
        ].shape[0]
        self.num_identifier = self.metadata[
            self.metadata["feature_type"] == FeatureType.IDENTIFIER
        ].shape[0]
        self.num_boolean = self.metadata[
            self.metadata["feature_type"] == FeatureType.BOOLEAN
        ].shape[0]

    def _render_key_metrics(self):
        """Render key metrics at the top of the overview card."""
        with st.container(horizontal=True):
            st.metric(label="Number of Rows", value=f"{self.data.shape[0]:,}")
            st.metric(label="Number of Columns", value=f"{self.data.shape[1]:,}")

    def _render_data_overview(self):
        """Render the data overview side panel."""
        with st.container(border=True, height="stretch"):
            st.markdown(
                f"""
            **Variable Types**  

            **Categorical**  
            Nominal: {self.num_nominal}  
            Ordinal: {self.num_ordinal}  
            Boolean: {self.num_boolean}  

            **Numerical**  
            Continuous: {self.num_continuous}  
            Discrete: {self.num_discrete}  

            **Others**  
            Identifier: {self.num_identifier}
            """
            )

    def _render_sample_data(self, option: DataSubset):
        """Render sample data based on the selected option."""
        match option:
            case DataSubset.FIRST_100:
                st.write(self.data.head(100))
            case DataSubset.RANDOM_SAMPLE:
                st.write(self.data.sample(100))
            case DataSubset.ALL:
                st.write(self.data)
            case _:
                # Should not reach here, but just in case
                st.error("Invalid option")

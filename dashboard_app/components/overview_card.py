from enum import Enum

import pandas as pd
import streamlit as st

from dashboard_app.constants.feature_type import FeatureType


class DataSubset(str, Enum):
    ALL = "All Data"
    FIRST_100 = "First 100 Rows"
    RANDOM_SAMPLE = "Random Sample"


class OverviewCard:

    def __init__(self, data: pd.DataFrame, metadata: pd.DataFrame):
        self.data = data
        self.metadata = metadata
        self._compute_summary()

    def _compute_summary(self):
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
        with st.container(horizontal=True):
            st.metric(label="Number of Rows", value=f"{self.data.shape[0]:,}")
            st.metric(label="Number of Columns", value=f"{self.data.shape[1]:,}")

    def render(self):

        # Data Overview
        with st.container():
            self._render_key_metrics()

            option = st.selectbox(
                "Select subset of raw data to view:",
                options=[member.value for member in DataSubset],
                index=0,
            )
            col1, col2 = st.columns([1, 5])
            with col2:
                # use a container for the data subset selection
                # select box should fit content

                if option == DataSubset.FIRST_100:
                    st.write(self.data.head(100))
                elif option == DataSubset.RANDOM_SAMPLE:
                    st.write(self.data.sample(100))
                elif option == DataSubset.ALL:
                    st.write(self.data)
            with col1:
                with st.container(border=True):
                    st.markdown(
                        f"""
                    Variable Types
                    **Identifier:** {self.num_identifier}
                    **Nominal:** {self.num_nominal}
                    **Ordinal:** {self.num_ordinal}
                    **Boolean:** {self.num_boolean}
                    **Continuous:** {self.num_continuous}
                    **Discrete:** {self.num_discrete}
                    """
                    )

            with st.expander("View Data Dictionary"):
                st.write(self.metadata)

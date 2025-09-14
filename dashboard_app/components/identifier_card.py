import pandas as pd
import streamlit as st
import plotly.express as px

from constants.feature_type import FeatureType


class IdentifierCard:
    _SUPPORTED_FEATURE_TYPES = [FeatureType.IDENTIFIER]

    def __init__(self, data: pd.Series, meta: pd.Series):
        self.data = data
        self.meta = meta
        self._compute_summary()

    @property
    def supported_feature_types(self) -> list[FeatureType]:
        return self._SUPPORTED_FEATURE_TYPES

    def is_supported(self, feature_type: FeatureType) -> bool:
        return feature_type in self.supported_feature_types

    def render(self):

        # Check if data can be visualised
        if not self.is_supported(self.feature_type):
            st.error(
                f"Feature type '{self.feature_type}' is not supported by {self.__class__.__name__}."
            )
            return

        # render the visualisations
        with st.container(border=True):
            col1, col2 = st.columns([5, 2])
            with col1:
                self._display_description()
            with col2:
                self._display_summary_table()

    def _compute_summary(self):
        self.feature_type: FeatureType = self.meta["feature_type"].values[0]
        self.variable_name: str = self.data.name
        self.description: str = self.meta["description"].values[0]
        self.category: str = self.meta["category"].values[0]
        self.count: int = self.data.count()
        self.missing: int = self.data.isnull().sum()
        self.unique: int = self.meta["n_unique"].values[0]
        self.mode: str = self.data.mode().values[0]
        self.mode_count: int = self.data.value_counts().values[0]
        self.duplicates: int = self.data.duplicated().sum()

    def _display_description(self):
        st.subheader(f"{self.variable_name}", divider=True)
        st.write(f"*category: {self.category}*")
        st.markdown(self.description)

    def _display_summary_table(self):
        with st.container(border=True):
            st.markdown(
                f"""**Feature Type:** {self.feature_type}  
            **Count:** {self.count:,}  
            **Missing Values\*:** {self.missing:,} *({self.missing/self.count:.1%})*  
            **Unique Values:** {self.unique:,}  
            **Mode:** {self.mode}  
            **Mode Count:** {self.mode_count:,}  
            **Duplicates:** {self.duplicates:,}  
            **missing values includes null values and placeholder '?'*
            """
            )

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy.stats import chi2_contingency, contingency

from dashboard_app.constants.feature_type import FeatureType

# TODO: need at least two types of visualisation? one for overiew and another
# interactive one for deep dive


class CategoricalCorrelationCard:

    def __init__(self, data: pd.DataFrame, metadata: pd.DataFrame):
        self.data = data
        self.metadata = metadata
        cat_feat_type = [
            FeatureType.NOMINAL,
            FeatureType.ORDINAL,
            FeatureType.BOOLEAN,
        ]
        self.categorical_vars = self.metadata[
            self.metadata["feature_type"].isin(cat_feat_type)
        ]["variable"]

    def _plot_stacked_barchart(self):
        var1 = st.selectbox("Select first variable (X-axis)", self.categorical_vars)
        var2 = st.selectbox(
            "Select second variable (Stacked color)", self.categorical_vars
        )

        # Do not allow selection of the same variable
        if var1 == var2:
            st.warning("Please select different variables.")
            return

        # Stacked bar chart showing proportions
        values = pd.crosstab(self.data[var1], self.data[var2], normalize="index")
        values = values.reset_index().melt(
            id_vars=var1, var_name=var2, value_name="Proportion"
        )
        values["text"] = values["Proportion"].apply(lambda x: f"{x:.1%}")

        fig = px.bar(
            values,
            x=var1,
            y="Proportion",
            color=var2,
            text="text",
            barmode="stack",
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_summary_description(self):
        st.subheader("Relationship Between Categorical Variables")
        st.markdown(
            """
            <TODO: add summary descriptions here>
        """
        )

    def render(self):
        with st.container(border=True):
            self._render_summary_description()
            self._plot_stacked_barchart()

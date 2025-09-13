import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go

from constants.feature_type import FeatureType
from constants.color import Color


class NumericalCategoricalCorrCard:
    def __init__(self, data: pd.DataFrame, metadata: pd.DataFrame):
        self.data = data
        self.metadata = metadata

    def render(self):
        with st.container(border=True):
            self._render_summary_description()
            self._plot_box_plot()

    def _render_summary_description(self):
        st.subheader("Relationship Between Numerical & Categorical Variables")

    def _plot_box_plot(self):
        cat_feat_type = [FeatureType.ORDINAL, FeatureType.NOMINAL, FeatureType.BOOLEAN]
        numerical_feat_type = [FeatureType.CONTINUOUS, FeatureType.DISCRETE]

        categorical_cols = self.metadata[
            self.metadata["feature_type"].isin(cat_feat_type)
        ]["variable"]
        numerical_cols = self.metadata[
            self.metadata["feature_type"].isin(numerical_feat_type)
        ]["variable"]

        cat_var = st.selectbox("Choose a categorical variable", categorical_cols)
        numerical_var = st.multiselect("Choose numerical variables", numerical_cols)

        if not numerical_var:
            st.warning("Please select at least one numerical variable.")
            return

        with st.container(
            horizontal=True,
            horizontal_alignment="right",
            vertical_alignment="center",
        ):
            st.text("Items per row:")
            num_cols_per_row = st.radio(
                "Number of columns per row",
                [1, 2, 3],
                key="num_cols_per_row",
                horizontal=True,
                label_visibility="collapsed",
                index=0,
            )
            show_outliers = st.toggle("Show Outliers", value=False)

        def gen_cols(count: int, num_cols: int = 3):
            num_rows = (count // num_cols) + (count % num_cols > 0)
            col_count = 0
            for _ in range(num_rows):
                cols = st.columns(num_cols)
                for col in cols:
                    if col_count < count:
                        yield col
                    col_count += 1

        for col, num_var in zip(
            gen_cols(len(numerical_var), num_cols_per_row), numerical_var
        ):
            with col:
                fig = px.box(
                    self.data,
                    y=cat_var,
                    x=num_var,
                    points="suspectedoutliers" if show_outliers else False,
                )
                fig.update_layout(margin=dict(t=30, b=30, l=0, r=0))
                fig.update_yaxes(tickfont=dict(size=14))
                fig.update_traces(
                    marker_color=Color.PRIMARY,
                    boxmean=True,
                )
                st.plotly_chart(fig, use_container_width=True)

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from constants.feature_type import FeatureType


class NumericalCorrelationCard:
    def __init__(self, data: pd.DataFrame, metadata: pd.DataFrame):
        self.data = data
        self.metadata = metadata
        self._compute_correlation_matrix()

    def _compute_correlation_matrix(self):
        numerical_feat_type = [FeatureType.CONTINUOUS, FeatureType.DISCRETE]
        self.numerical_features = self.metadata[
            self.metadata["feature_type"].isin(numerical_feat_type)
        ]["variable"]
        self.corr_matrix = self.data[self.numerical_features].corr(method="spearman")

    def _plot_pairwise_scatter(self):

        def add_jitter(series, scale=0.1):
            jitter = np.random.uniform(-scale, scale, size=len(series)) * (
                series.max() - series.min()
            )
            return series + jitter

        st.markdown("### Pairwise Scatter Plot")
        # Use selectbox
        first_feat = st.selectbox(
            "Select first variable (X-axis)", self.numerical_features
        )
        second_feat = st.selectbox(
            "Select second variable (Y-axis)", self.numerical_features
        )

        if first_feat == second_feat:
            st.warning("Please select two different variables.")
            return

        corr_coeff = self.corr_matrix.loc[first_feat, second_feat]

        first_desc = self.metadata[self.metadata["variable"] == first_feat][
            "description"
        ].values[0]
        second_desc = self.metadata[self.metadata["variable"] == second_feat][
            "description"
        ].values[0]

        col1, col2 = st.columns([2, 6])
        with col1:
            with st.container(border=True):
                st.metric(
                    """Spearman Correlation
                    Coefficient""",
                    f"{corr_coeff:.2f}",
                )

        with col2:
            st.markdown(
                f"""
                **Pairwise correlation Analysis:**

                **{first_feat}**
                    {first_desc}

                **{second_feat}**
                    {second_desc}
                """
            )
        jitter = st.slider("**Jitter amount**", 0.0, 1.0, 0.5)
        data_jittered = self.data[self.numerical_features].copy()
        for col in self.numerical_features:
            data_jittered[col] = add_jitter(data_jittered[col], scale=jitter)

        fig = px.scatter_matrix(
            # self.data,
            data_jittered,
            dimensions=[first_feat, second_feat],
            # title="Pairwise Scatterplot",
            height=600,
            width=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    def _plot_correlation_matrix(self):
        fig = px.imshow(
            self.corr_matrix.round(2),
            text_auto=True,
            color_continuous_scale="RdBu_r",
            width=800,
        )

        st.plotly_chart(fig, use_container_width=True)

    def _plot_correlation_matrix(self):
        st.markdown("### Correlation Heatmap (Spearman)")
        fig = px.imshow(
            self.corr_matrix.round(2),
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            width=600,
            height=600,
            # title="Correlation Heatmap (Spearman)",
        )
        st.plotly_chart(fig, use_container_width=True)

    def _render_summary_description(self):
        # Describe the correlation analysis
        st.subheader("Relationship Between Numerical Variables")
        st.markdown(
            f"""
            **Spearman correlation coefficient** is used to measure the correlation 
            between the numerical variables in the dataset. It uses the rank values 
            instead of the raw values, making it suitable for the dataset as the 
            numerical values are all of a discrete nature.  

            It **ranges from -1 to 1**, measuring the strength and direction of monotonic relationships  
            - **As the correlation coefficient approaches 1**, it indicates a strong positive correlation, meaning that as one variable increases, the other variable tends to also increase.
            - **As the correlation coefficient approaches -1**, it indicates a strong negative correlation, meaning that as one variable increases, the other variable tends to decrease.
            - **A correlation coefficient around 0** suggests no monotonic relationship between the variables.  

        """
        )

    def render(self):
        with st.container(border=True):
            self._render_summary_description()
            self._plot_correlation_matrix()
            self._plot_pairwise_scatter()

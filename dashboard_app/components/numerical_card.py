import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

from dashboard_app.constants.feature_type import FeatureType


class NumericalCard:
    _SUPPORTED_FEATURE_TYPES = [FeatureType.DISCRETE, FeatureType.CONTINUOUS]

    def __init__(self, data: pd.DataFrame, metadata: pd.DataFrame):
        self.data = data
        self.metadata = metadata
        self._compute_summary()

    @property
    def supported_feature_types(self) -> list[FeatureType]:
        return self._SUPPORTED_FEATURE_TYPES

    def is_supported(self, feature_type: FeatureType) -> bool:
        return feature_type in self.supported_feature_types

    def render(self):
        feature_type = self.feature_type
        if not self.is_supported(self.feature_type):
            st.warning(f"Feature type '{feature_type}' is not supported.")
            return

        with st.container(border=True):
            match self.feature_type:
                case FeatureType.DISCRETE:
                    col1, col2 = st.columns([5, 2])
                    with col1:
                        self._display_description()
                        self._plot_box_plot()
                    with col2:
                        self._display_summary_table()
                    self._plot_bar_chart()
                case FeatureType.CONTINUOUS:
                    col1, col2 = st.columns([5, 2])
                    with col1:
                        self._display_description()
                        self._plot_box_plot()
                    with col2:
                        self._display_summary_table()
                    self._plot_histogram()

    def _compute_summary(self):

        self.variable_name = self.data.name
        self.feature_type = self.metadata["feature_type"].values[0]
        self.description = self.metadata["description"].values[0]
        self.category = self.metadata["category"].values[0]
        self.count = self.data.count()
        self.missing: int = self.metadata["missing_values"].values[0]
        self.unique = self.metadata["n_unique"].values[0]
        self.mean = self.data.mean()
        self.std_err = self.data.std()
        self.min_val = self.data.min()
        self.max_val = self.data.max()
        self.q1 = self.data.quantile(0.25)
        self.median = self.data.quantile(0.5)
        self.q3 = self.data.quantile(0.75)
        self.iqr = self.q3 - self.q1
        self.lower_outliers = (self.data < (self.q1 - 1.5 * self.iqr)).sum()
        self.upper_outliers = (self.data > (self.q3 + 1.5 * self.iqr)).sum()
        self.num_outliers = self.lower_outliers + self.upper_outliers

    def _display_description(self):
        st.subheader(f"{self.variable_name}", divider=True)
        st.write(f"*category: {self.category}*")
        st.markdown(self.description)

    def _display_summary_table(self):
        with st.container(border=True):
            st.markdown(
                f"""**Data Type:** {self.feature_type}  
            **Count:** {self.count:,}  
            **Missing Values\*:** {self.missing:,} *({self.missing/self.count:.1%})*  
            **Unique Values:** {self.unique:,}  
            **Mean:** {self.mean:,.2f}  
            **Standard Deviation:** {self.std_err:,.2f}  
            **Min:** {self.min_val:,.2f}  
            **Max:** {self.max_val:,.2f}  
            **25% Percentile:** {self.q1:,.2f}  
            **50% Percentile:** {self.median:,.2f}  
            **75% Percentile:** {self.q3:,.2f}  
            **Lower Outliers:** {self.lower_outliers:,}  
            **Upper Outliers:** {self.upper_outliers:,}  
            **Total Outliers:** {self.num_outliers:,}  
            **missing values includes null values and placeholder '?'*
            """
            )

    def _plot_box_plot(self):
        # --- Box plot --- #
        fig = px.box(
            x=self.data,
            orientation="h",
            title=f"Box Plot of {self.variable_name}",
            labels={"x": self.variable_name},
            width=800,
            height=300,
        )

        # Add annotations for min, max, 25%, median, 75%
        fig.add_annotation(
            x=self.min_val - 0.7,
            y=0.2,
            text=f"Min: {self.min_val:,.2f}",
            font=dict(size=12),
            showarrow=False,
        )
        fig.add_annotation(
            x=self.q1,
            y=0.3,
            text=f"25%: {self.q1:,.2f}",
            font=dict(size=12),
            showarrow=False,
            ax=-30,
            ay=0,
        )
        fig.add_annotation(
            x=self.median,
            y=0.4,
            text=f"Median: {self.median:,.2f}",
            font=dict(size=12),
            showarrow=False,
            ax=-30,
            ay=0,
        )
        fig.add_annotation(
            x=self.q3,
            y=0.3,
            text=f"75%: {self.q3:,.2f}",
            font=dict(size=12),
            showarrow=False,
            ax=-30,
            ay=0,
        )
        fig.add_annotation(
            x=self.max_val + 0.5,
            y=0.2,
            text=f"Max: {self.max_val:,.2f}",
            font=dict(size=12),
            showarrow=False,
        )

        # Show the boxplot
        st.plotly_chart(fig)

    def _plot_histogram(self):
        # Calculate KDE
        kde = gaussian_kde(self.data)
        x_vals = np.linspace(self.min_val, self.max_val, 1000)
        kde_vals = kde(x_vals)

        # Create figure
        fig = go.Figure()

        # Add histogram
        fig.add_trace(
            go.Histogram(
                x=self.data,
                histnorm="probability density",
                name="Histogram",
                opacity=0.6,
                marker_color="lightblue",
            )
        )

        # Add KDE line
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=kde_vals,
                mode="lines",
                name="KDE",
                line=dict(color="darkblue", width=2),
            )
        )

        # Layout
        fig.update_layout(
            title=f"Histogram + KDE for {self.variable_name}",
            xaxis_title=self.variable_name,
            yaxis_title="Density",
            barmode="overlay",
        )

        st.plotly_chart(fig)

    def _plot_bar_chart(self):
        category_counts = self.data.value_counts()
        category_perc = self.data.value_counts(normalize=True)

        # Show category counts and percentages for each bar
        column_text = [
            f"{count:,} ({perc:.1%})"
            for count, perc in zip(category_counts.values, category_perc.values)
        ]
        fig = px.bar(
            x=category_counts.index,
            y=category_counts.values,
            labels={"x": "Categories", "y": "Count"},
            title=f"Distribution of {self.variable_name}",
            text=column_text,
            width=800,
            height=300,
        )

        st.plotly_chart(fig)

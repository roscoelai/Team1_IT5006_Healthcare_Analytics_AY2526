import pandas as pd
import streamlit as st
import plotly.express as px

from constants.feature_type import FeatureType
from constants.color import Color


class CategoricalCard:
    # TODO: for ordinal data, need to preserve order

    _SUPPORTED_FEATURE_TYPES = [
        FeatureType.NOMINAL,
        FeatureType.ORDINAL,
        FeatureType.BOOLEAN,
    ]

    def __init__(self, data: pd.Series, metadata: pd.Series):
        self.data = data
        self.metadata = metadata
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

            match self.feature_type:
                case FeatureType.NOMINAL | FeatureType.ORDINAL:
                    # NOTE: bar chart clearer for nominal/ordinal features
                    col1, col2 = st.columns([5, 2])
                    with col1:
                        self._display_description()
                    with col2:
                        self._display_summary_table()
                    self._plot_bar_chart()
                case FeatureType.BOOLEAN:
                    # Use pie chart for boolean features
                    self._display_description()
                    col1, col2 = st.columns([5, 2])
                    with col1:
                        self._plot_pie_chart()
                    with col2:
                        self._display_summary_table()
                case _:
                    st.warning("Unsupported feature type.")

    def _compute_summary(self):
        self.feature_type: FeatureType = self.metadata["feature_type"].values[0]
        self.variable_name: str = self.data.name
        self.description: str = self.metadata["description"].values[0]
        self.category: str = self.metadata["category"].values[0]
        self.count: int = self.data.count()
        self.missing: int = self.metadata["missing_values"].values[0]
        self.unique: int = self.metadata["n_unique"].values[0]
        self.mode: str = self.data.mode().values[0]
        self.mode_count: int = self.data.value_counts().values[0]

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
            **missing values includes null values and placeholder '?'*
            """
            )

    def _plot_bar_chart(self):
        data_series = self.data.sort_values()
        category_counts = data_series.value_counts()
        category_perc = data_series.value_counts(normalize=True)

        # Show category counts and percentages for each bar
        column_text = [
            f"{count:,} ({perc:.1%})"
            for count, perc in zip(category_counts.values, category_perc.values)
        ]
        fig = px.bar(
            x=category_counts.index,
            y=category_counts.values,
            labels={"x": self.variable_name, "y": "Count"},
            title=f"Distribution of {self.variable_name}",
            text=column_text,
            width=800,
            height=300,
        )

        fig.update_traces(
            textposition="auto",
            textfont_size=14,
            textfont_color=Color.TEXT,
            marker_color=Color.PRIMARY,
        )

        fig.update_layout(margin=dict(t=40, b=40, l=0, r=0))
        st.plotly_chart(fig)

    def _plot_pie_chart(self):
        data_series = self.data.sort_values()
        category_counts = data_series.value_counts()
        fig = px.pie(
            names=category_counts.index,
            values=category_counts.values,
            title=f"Distribution of {self.variable_name}",
            width=400,
            height=300,
        )
        fig.update_traces(textinfo="label+percent+value", textposition="inside")
        fig.update_layout(
            showlegend=False,
            margin=dict(t=40, b=40, l=0, r=0),
        )
        fig.update_traces(
            marker=dict(colors=[Color.PRIMARY, Color.SECONDARY]),
            textfont_size=14,
            textfont_color=Color.TEXT,
        )
        st.plotly_chart(fig)

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go

from dashboard_app.constants.feature_type import FeatureType

# bar plot groupby for numerical <-> ordinal grp by categorical, plot mean of numerical
# box plot for numerical <-> ordinal


class TargetCorrCard:

    def __init__(self, data: pd.DataFrame, metadata: pd.DataFrame, target_var: str):
        self.df = data
        self.metadata = metadata
        self.target_var = target_var

    def _render_summary_description(self):
        st.subheader(f"**Target Variable**: {self.target_var}")
        target_desc = self.metadata[self.metadata["variable"] == self.target_var][
            "description"
        ].values[0]
        st.markdown(f"{target_desc}")
        st.subheader("Key Findings:")
        st.markdown(
            """
        1. **Older patients display higher readmission rates**, with those aged 70 and 
        above particularly prone to short-term (<30 days) readmissions. This is 
        expected due to the higher risk of health complications in older adults.  
        2. **Longer hospital stay seems to be correlated with higher likelihood of 
        being readmitted**. Patients discharged after extended stays may have unresolved 
        complications or more severe baseline conditions, increasing their subsequent 
        risk of readmission.  
        3. **Patients with a higher number of medications appear to have a higher likelihood of readmission.**
        4. **Patients on diabetic medications also have a higher likelihood of readmission.**
        5. **Patients with frequent adjustments to their diabetes medications, particularly insulin, 
        are more likely to be readmitted within 30 days.** This suggests unstable 
        glycemic control and difficulty in stabilizing treatment regimens. The presence of polypharmacy may also serve as a proxy for overall health complexity.
        6. **African American and Caucasian patients show higher readmission rates.** Missing values 
        in this feature weaken the confidence of the finding, but it nonetheless raises questions 
        about the potential diet / healthcare support in these countries that may affect the readmission rates.
        """
        )

    def _plot_boxplot(self):
        readmitted_order = ["<30", ">30", "NO"]
        readmitted_color = ["#FF5733", "#FFA500", "#006400"]
        numerical_cols = self.metadata[
            self.metadata["feature_type"].isin(
                [FeatureType.CONTINUOUS, FeatureType.DISCRETE]
            )
        ]["variable"]
        num_var = st.multiselect(
            "Choose numerical variables",
            numerical_cols,
            key="target_num_var",
            default=numerical_cols,
        )

        if not num_var:
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
                key="target_cols_per_row",
                horizontal=True,
                label_visibility="collapsed",
                index=1,
            )
            show_outliers = st.toggle(
                "Show Outliers", value=False, key="target_show_outliers"
            )

        def gen_cols(count: int, num_cols: int = 3):
            num_rows = (count // num_cols) + (count % num_cols > 0)
            col_count = 0
            for _ in range(num_rows):
                cols = st.columns(num_cols)
                for col in cols:
                    if col_count < count:
                        yield col
                    col_count += 1

        for col, num_var in zip(gen_cols(len(num_var), num_cols_per_row), num_var):
            with col:
                fig = px.box(
                    self.df,
                    x=num_var,
                    y=self.target_var,
                    points="suspectedoutliers" if show_outliers else False,
                    color="readmitted",
                    category_orders={self.target_var: readmitted_order},
                    color_discrete_sequence=readmitted_color,
                )
                fig.update_yaxes(tickfont=dict(family="Arial Black", size=12))
                fig.update_layout(showlegend=False)
                st.plotly_chart(
                    fig, key=f"target_box_{num_var}", use_container_width=True
                )

    def _plot_barchart(self):
        categorical_cols = self.metadata[
            self.metadata["feature_type"].isin(
                [FeatureType.NOMINAL, FeatureType.ORDINAL, FeatureType.BOOLEAN]
            )
        ]["variable"]
        medication_vars = self.metadata[self.metadata["category"] == "Medications"][
            "variable"
        ]
        categorical_cols = categorical_cols[~categorical_cols.isin(medication_vars)]
        cat_var = st.multiselect(
            "Choose categorical variables",
            categorical_cols,
            key="target_cat_var",
        )

        if not cat_var:
            st.warning("Please select at least one categorical variable.")
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
                key="target_cat_cols_per_row",
                horizontal=True,
                label_visibility="collapsed",
                index=1,
            )

        def gen_cols(count: int, num_cols: int = 3):
            num_rows = (count // num_cols) + (count % num_cols > 0)
            col_count = 0
            for _ in range(num_rows):
                cols = st.columns(num_cols)
                for col in cols:
                    if col_count < count:
                        yield col
                    col_count += 1

        for col, cat_var in zip(gen_cols(len(cat_var), num_cols_per_row), cat_var):
            with col:
                # Grouped bar chart showing %
                values = (
                    pd.crosstab(
                        self.df[cat_var],
                        self.df[self.target_var],
                        normalize="index",
                    )
                    .reset_index()
                    .melt(
                        id_vars=cat_var,
                        var_name=self.target_var,
                        value_name="Proportion",
                    )
                )
                values["text"] = values["Proportion"].apply(lambda x: f"{x:.1%}")
                readmitted_order = ["<30", ">30", "NO"]
                readmitted_color = ["#FF5733", "#CCCC00", "#98FB98"]
                fig = px.bar(
                    values,
                    x=cat_var,
                    y="Proportion",
                    color=self.target_var,
                    text="text",
                    barmode="group",
                    category_orders={self.target_var: readmitted_order},
                    color_discrete_sequence=readmitted_color,
                    range_y=[0, 1],
                )
                fig.update_yaxes(tickfont=dict(family="Arial Black", size=12))
                st.plotly_chart(
                    fig, key=f"target_bar_{cat_var}", use_container_width=True
                )

    def render(self):
        with st.container(border=True):
            self._render_summary_description()

            tab1, tab2, tab3 = st.tabs(
                [
                    "Target <-> Numerical",
                    "Target <-> Categorical (Excl Medications)",
                    "Target <-> Categorical (Medications)",
                ]
            )

            with tab1:
                self._plot_boxplot()
            with tab2:
                self._plot_barchart()

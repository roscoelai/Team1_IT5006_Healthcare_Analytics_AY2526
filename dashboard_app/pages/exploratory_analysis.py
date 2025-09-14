import re
import os
import sys
from enum import Enum

import pandas as pd
import streamlit as st
import plotly.express as px

from components.categorical_card import CategoricalCard
from components.numerical_card import NumericalCard
from components.identifier_card import IdentifierCard
from components.num_corr_card import NumericalCorrelationCard
from components.cat_corr_card import CategoricalCorrelationCard
from components.num_cat_corr_card import NumericalCategoricalCorrCard
from components.target_corr_card import TargetCorrCard
from components.overview_card import OverviewCard
from constants.feature_type import FeatureType
from constants.color import Color
from utils.dataloader import DataLoader

# Load data required for the page
df = DataLoader.get_data()
metadata = DataLoader.get_metadata()

# ---
# TODO: check if all the .copy() are necessary - not needed if not modifying the df
# TODO: break down into card components

st.title("Exploratory Data Analysis")

# ------------------------------------------------------------------------------
# Univariate Analysis
# ------------------------------------------------------------------------------
st.subheader("Univariate Analysis")
st.markdown(
    """
    Univariate analysis involves examining the distribution and characteristics of
    **individual variables** in the dataset.  

    Key findings from the univariate analysis include:  
"""
)


# with st.expander("Missing Values in Features", expanded=False):
with st.container(border=True):
    col1, col2 = st.columns([2, 5])
    with col1:
        st.subheader("Missing Values in Features")
        st.markdown(
            """
                - There are **ten** features with missing values.  
                - `weight` has too many missing values to be useful and will 
                be dropped from further analysis.  
                - The other features with missing values will need to be handled
                appropriately during preprocessing.
            """
        )

    with col2:
        # Visualization
        missing_df = (
            metadata[metadata["missing_values"] > 0]
            .sort_values(by="missing_values", ascending=False)
            .reset_index(drop=True)
        )

        # percentage of missing values
        missing_df["missing_perc"] = (
            missing_df["missing_values"] / len(df) * 100
        ).round(2)

        fig = px.bar(
            missing_df[missing_df["missing_values"] > 0].sort_values(
                by="missing_perc", ascending=False
            ),
            x="variable",
            y="missing_perc",
            text="missing_perc",
            labels={
                "variable": "Feature",
                "missing_perc": "Percentage of Missing Values",
            },
            title="Percentage of Missing Values by Feature",
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="auto",
            textfont_size=14,
            textfont_color=Color.TEXT,
            marker_color=Color.DANGER,
        )

        # Draw a vertical line seperating weight from other features
        fig.add_vline(
            x=0.5,
            line_width=3,
            line_dash="dash",
            line_color=Color.TEXT,
        )

        fig.data[0].marker.color = [
            Color.DANGER if var == "weight" else Color.MUTED
            for var in missing_df[missing_df["missing_values"] > 0]["variable"]
        ]
        # fig.data[0].marker.opacity = [
        #     0.3 if var == "weight" else 1.0
        #     for var in missing_df[missing_df["missing_values"] > 0]["variable"]
        # ]
        fig.data[0].marker.pattern.shape = [
            "/" if var == "weight" else ""
            for var in missing_df[missing_df["missing_values"] > 0]["variable"]
        ]

        # Add legend to state that remaining features will need to be handled
        fig.add_annotation(
            x=0.2,
            y=0.95,
            xref="paper",
            yref="paper",
            text="<b>   Weight</b> will be dropped.<br>Remaining features will be handled<br>during preprocessing.",
            showarrow=False,
            font=dict(color=Color.ANNOTATION_TEXT, size=14),
            bgcolor=Color.ANNOTATION_BG,
            bordercolor=Color.BORDER,
            borderpad=5,
        )

        st.plotly_chart(fig, use_container_width=True)

# with st.expander("Uniformity in Features", expanded=False):
with st.container(border=True):
    col1, col2 = st.columns([2, 5])

    with col1:
        st.subheader("Features with Low Variance")
        st.markdown(
            """
            - There are two features with no variance *(i.e., the same value for all records)* 
            and a total of 25 features with >80% uniformity.  
            - These need to be assessed futher to determine if they are informative or
            should be removed from further analysis.  
            
            `% mode` - The proportion the mode value contributes to the total
            """
        )
    with col2:
        mode_info = df.apply(
            lambda x: pd.Series(
                {
                    "mode": x.mode()[0],
                    "% Mode": x.value_counts(normalize=True).max(),
                }
            )
        ).T.reset_index()
        mode_info.columns = ["variable", "mode", "% mode"]
        mode_info["mode"] = mode_info["mode"].apply(
            lambda v: int(v) if isinstance(v, float) and v.is_integer() else v
        )

        threshold = 0.80  # filter out those with mode perc > threshold
        low_variance_df = (
            mode_info[mode_info["% mode"] > threshold]
            .sort_values(by="% mode", ascending=False)
            .reset_index(drop=True)
        )

        # merge with metadata
        low_variance_df = low_variance_df.merge(
            metadata[["variable", "feature_type", "category", "description"]],
            on="variable",
            how="left",
        )

        low_variance_df.index += 1  # Start index from 1 instead of 0
        st.write(
            f"*{low_variance_df.shape[0]:,} features with > {threshold:.0%} uniformity*"
        )
        low_variance_df["% mode"] = low_variance_df["% mode"].apply(
            lambda x: f"{x:.3%}"
        )

        # gradient highlight % mode column from 80% (light yellow) to 100% (dark orange)
        def highlight_mode_perc(s):
            # use background gradient
            return [
                (
                    f"background-color: {Color.DANGER.value}"
                    if float(v.strip("%")) >= 95
                    else (
                        f"background-color: {Color.WARNING.value}"
                        if float(v.strip("%")) >= 90
                        else (
                            f"background-color: {Color.HIGHLIGHT.value}"
                            if float(v.strip("%")) >= 80
                            else ""
                        )
                    )
                )
                for v in s
            ]

        st.write(low_variance_df.style.apply(highlight_mode_perc, subset=["% mode"]))


# with st.expander("Multiple Patient Encounters", expanded=False):
with st.container(border=True):
    col1, col2 = st.columns([2, 5])
    with col1:
        st.subheader("Multiple Patient Encounters")
        st.markdown(
            """
            - There are no duplicate records (patient encounters) in the dataset  
            - However, **16,773 of the 71,518 unique patients** have multiple hospital encounters.  
            - Of the patients with multiple encounters, the **majority (62.2%) had only 2 encounters**  
            - This will need to be taken into account during analysis, as it may affect the independence
            of observations.  
            """
        )
    with col2:
        tab1, tab2 = st.tabs(["Proportion of Patients", "Number of Encounters"])
        with tab1:
            patient_encounter_counts = df["patient_nbr"].value_counts()
            multiple_encounters = (patient_encounter_counts > 1).sum()
            total_unique_patients = patient_encounter_counts.shape[0]

            fig = px.pie(
                names=["Multiple Encounters", "Single Encounter"],
                values=[
                    multiple_encounters,
                    total_unique_patients - multiple_encounters,
                ],
                title="Proportion of Patients with Multiple Encounters",
                color=["Multiple Encounters", "Single Encounter"],
                color_discrete_map={
                    "Multiple Encounters": Color.SECONDARY,
                    "Single Encounter": Color.PRIMARY,
                },
                height=450,
            )

            fig.update_traces(
                marker=dict(line=dict(color=Color.BORDER, width=1)),
                textposition="auto",
                textinfo="percent+label+value",
                textfont_size=14,
                textfont_color=Color.TEXT,
            )
            fig.update_layout(
                showlegend=False,
                margin=dict(t=50, b=50, l=0, r=0),
            )
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            # bin the number of encounters
            encounter_bins = [2, 3, 5, 10, 20]
            encounter_labels = ["2", "3-4", "5-9", "10-19", "20+"]
            binned_encounters = pd.cut(
                patient_encounter_counts[patient_encounter_counts > 1],
                bins=[1] + encounter_bins,
                labels=encounter_labels,
                right=True,
            )
            binned_counts = binned_encounters.value_counts().sort_index()
            binned_perc = (binned_counts / binned_counts.sum()).round(4)
            binned_text = [
                f"{count:,} ({perc:.1%})"
                for count, perc in zip(binned_counts.values, binned_perc.values)
            ]
            fig = px.bar(
                x=binned_counts.index,
                y=binned_counts.values,
                text=binned_text,
                labels={"x": "Number of Encounters", "y": "Count of Patients"},
                title="Distribution of Number of Encounters for Patients with Multiple Encounters",
            )
            fig.update_traces(
                textposition="auto",
                textfont_size=14,
                textfont_color=Color.TEXT,
            )

            fig.data[0].marker.color = [
                Color.DANGER if i == 0 else Color.MUTED
                for i in range(len(binned_counts))
            ]

            fig.add_annotation(
                x=0.2,
                y=0.95,
                xref="paper",
                yref="paper",
                text=f"Of the <b>{multiple_encounters:,} patients</b> with multiple encounters, <br>the majority <b>({binned_perc[0]:.1%}) had only 2 encounters.</b>",
                showarrow=False,
                font=dict(color=Color.ANNOTATION_TEXT, size=14),
                bgcolor=Color.ANNOTATION_BG,
                bordercolor=Color.BORDER,
                borderpad=5,
            )

            st.plotly_chart(fig, use_container_width=True)


# with st.expander("Outliers", expanded=False):
with st.container(border=True):
    col1, col2 = st.columns([2, 5])
    with col1:
        st.subheader("Features with Outliers")
        st.markdown(
            """
            - Several numerical features exhibit outliers (beyond 1.5\*IQR from Q1 and Q3). 
            and some have extreme outliers (beyond 3\*IQR).  
            - These outliers will need to be addressed during preprocessing to 
            prevent them from skewing analysis
            results.  
        """
        )
    with col2:
        # Plot boxplots for numerical features with outliers
        numerical_features = metadata[
            metadata["feature_type"].isin(
                [FeatureType.DISCRETE, FeatureType.CONTINUOUS]
            )
        ]["variable"].tolist()

        # number of outliers defined as those beyond 1.5*IQR from Q1 and Q3
        def count_outliers(series: pd.Series, multiplier: float = 1.5) -> int:
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - multiplier * iqr
            upper_bound = q3 + multiplier * iqr
            return ((series < lower_bound) | (series > upper_bound)).sum()

        # OUtlier dataframe
        outlier_info = []
        for feature in numerical_features:
            series = df[feature]
            num_outliers_1_5 = count_outliers(series, 1.5)
            num_outliers_3_0 = count_outliers(series, 3.0)
            perc_outliers_1_5 = num_outliers_1_5 / series.count()
            perc_outliers_3_0 = num_outliers_3_0 / series.count()
            outlier_info.append(
                {
                    "variable": feature,
                    "Outliers_15_IQR": num_outliers_1_5,
                    "Outliers_15_IQR_PERC": perc_outliers_1_5 * 100,
                    "Outliers_30_IQR": num_outliers_3_0,
                    "Outliers_30_IQR_PERC": perc_outliers_3_0 * 100,
                }
            )

        outlier_df = pd.DataFrame(outlier_info)
        outlier_df = outlier_df.sort_values(
            by="Outliers_15_IQR", ascending=False
        ).reset_index(drop=True)

        outlier_df["Outliers (1.5 IQR)"] = outlier_df["Outliers_15_IQR"].apply(
            lambda x: f"{x:,}"
        ) + outlier_df["Outliers_15_IQR_PERC"].apply(lambda x: f" ({x:.2f}%)")
        outlier_df["Outliers (3.0 IQR)"] = outlier_df["Outliers_30_IQR"].apply(
            lambda x: f"{x:,}"
        ) + outlier_df["Outliers_30_IQR_PERC"].apply(lambda x: f" ({x:.2f}%)")

        outlier_df.index += 1  # Start index from 1 instead of 0

        def highlight_outlier_perc(s):
            # opacity from 0 to 1 based on percentage
            opacity = [
                min(1, float(v) / 20)
                for v in s
                for v in re.findall(r"\((\d+\.\d+)%\)", v)
            ]
            return [f"background-color: rgba(255, 0, 0, {o})" for o in opacity]

        st.write(f"Number of records in dataset: {len(df):,}")
        st.write(
            # outlier_df,
            outlier_df[
                ["variable", "Outliers (1.5 IQR)", "Outliers (3.0 IQR)"]
            ].style.apply(
                highlight_outlier_perc,
                subset=["Outliers (1.5 IQR)", "Outliers (3.0 IQR)"],
            )
        )

# with st.expander("Distribution of Demographic Features", expanded=False):
with st.container(border=True):
    col1, col2 = st.columns([2, 5])
    with col1:
        st.subheader("Distribution of Demographic Features")
        st.markdown(
            """
            - Relatively low number of encounters involving patients below 30 years old.  
            - High number of encounters involving older patients, peaking at [70-80) age group.  
            - The drop in encounters involving patients above 80 years old is likely due to mortality.
            - Fairly balanced distribution of encounters involving male and female patients.  
            - Racial distribution heavily skewed towards Caucasian and African American patients,
            which together account for over 93.7% of the records.
            """
        )
    with col2:
        tab1, tab2, tab3 = st.tabs(["Age", "Gender", "Race"])
        with tab1:
            # filter only readmitted patients
            age_df = df.copy()
            # plot bar chart with x age group
            fig = px.bar(
                age_df["age"].value_counts().sort_index(),
                labels={"index": "Age Group", "value": "Count"},
                title="Distribution of Age Groups",
            )

            # Add % text on top of each bar
            age_perc = age_df["age"].value_counts(normalize=True).sort_index().round(4)
            age_text = [
                f"{count:,} ({perc:.1%})"
                for count, perc in zip(
                    age_df["age"].value_counts().sort_index(), age_perc
                )
            ]
            fig.update_traces(
                text=age_text,
                textposition="auto",
                textfont_size=16,
                marker_color=Color.PRIMARY,
            )
            fig.update_layout(
                yaxis=dict(range=[0, age_df["age"].value_counts().max() * 1.1]),
                showlegend=False,
                margin=dict(t=50, b=50, l=0, r=0),
            )

            fig.add_vline(
                x=2.5,
                line_width=3,
                line_dash="dash",
                line_color=Color.TEXT,
            )
            fig.add_vline(
                x=7.5,
                line_width=3,
                line_dash="dash",
                line_color=Color.TEXT,
            )

            fig.add_annotation(
                x=0.55,
                y=1,
                xref="paper",
                yref="paper",
                text="More encounters involves<br> older patients, peaking<br> at [70-80) age group.",
                showarrow=False,
                font=dict(color=Color.ANNOTATION_TEXT, size=14),
                bgcolor=Color.ANNOTATION_BG,
                bordercolor=Color.BORDER,
                borderpad=5,
            )
            fig.add_annotation(
                x=0,
                y=0.2,
                xref="paper",
                yref="paper",
                text="Relatively low encounters<br> involving patients below 30",
                showarrow=False,
                font=dict(color=Color.ANNOTATION_TEXT, size=14),
                bgcolor=Color.ANNOTATION_BG,
                bordercolor=Color.BORDER,
                borderpad=5,
            )
            fig.add_annotation(
                x=0.9,
                y=0.1,
                xref="paper",
                yref="paper",
                text="Lower encounters<br> for patients above 80,<br> likely due to mortality",
                showarrow=False,
                font=dict(color=Color.ANNOTATION_TEXT, size=14),
                bgcolor=Color.ANNOTATION_BG,
                bordercolor=Color.BORDER,
                borderpad=5,
            )

            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            gender_df = df.copy()
            # pie chart showing distribution of gender
            fig = px.pie(
                gender_df,
                names="gender",
                title="Distribution of Gender",
                labels={"gender": "Gender"},
                height=450,
                color="gender",
                color_discrete_map={
                    "Male": Color.PRIMARY,
                    "Female": Color.SECONDARY,
                },
            )
            fig.update_traces(
                textinfo="percent+label+value",
                textposition="auto",
                textfont_size=16,
                marker=dict(line=dict(color=Color.BORDER, width=2)),
            )
            fig.update_layout(
                showlegend=False,
                margin=dict(t=70, b=70, l=0, r=0),
            )
            st.plotly_chart(fig, use_container_width=True)
        with tab3:
            race_df = df.copy()
            fig = px.treemap(
                race_df,
                path=["race"],
                title="Racial Distribution",
            )
            fig.update_layout(margin=dict(t=50, b=50, l=0, r=0))
            fig.update_traces(
                textinfo="percent entry+label",
                textposition="middle center",
                textfont_size=16,
                marker=dict(line=dict(color=Color.BORDER, width=2)),
            )
            # show % in hover
            fig.data[0].hovertemplate = (
                "<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percentParent:.1%}<extra></extra>"
            )
            st.plotly_chart(fig, use_container_width=True)

# with st.expander(
#     "Distribution of Readmission Status (Target Variable)", expanded=False
# ):
with st.container(border=True):
    col1, col2 = st.columns([2, 5])
    with col1:
        # Text Summary
        st.subheader("Moderate class imbalance in readmission status")
        st.markdown(
            """
            - `NO` (no readmission) accounts for over half of the records *(53.9%)*  
            - It ***might be*** beneficial to group the readmission status into
            a binary classification problem (e.g., "No Readmission" vs. 
            "Readmission") to address the class imbalance.  
            """
        )

    with col2:
        # Visualization
        category_perc = df["readmitted"].value_counts(normalize=True)
        count = df["readmitted"].value_counts()
        column_text = [
            f"{count:,} ({percent:.1%})"
            for count, percent in zip(count.values, category_perc.values)
        ]

        color_map = {"NO": Color.SUCCESS, ">30": Color.MUTED, "<30": Color.MUTED}
        colors = [color_map[val] for val in category_perc.index]

        # Bar chart y showing count, different colors for each bar
        fig = px.bar(
            x=category_perc.index,
            y=count.values,
            text=column_text,
            labels={"x": "Readmission Status", "y": "Count"},
            title="Distribution of Readmission Status",
            color=category_perc.index,
            color_discrete_map=color_map,
        )

        fig.update_traces(
            textposition="auto",
            textfont_size=16,
            textfont_color=Color.TEXT,
        )

        # Highlight the "NO" bar with an annotation
        fig.add_annotation(
            x="NO",
            y=count["NO"],
            text='"NO" has more than half of the records',
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowcolor=Color.SUCCESS,
            bgcolor=Color.ANNOTATION_BG,
            bordercolor=Color.BORDER,
            borderpad=5,
            font=dict(color=Color.ANNOTATION_TEXT, size=14),
            ax=0,
            ay=-40,
        )

        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

with st.expander("Interactive Analysis (Univariate)", expanded=False):
    st.info(
        """🔍 Select a variable from the dropdown below to view its univariate 
        metrics and visualizations."""
    )
    selected_var = st.selectbox(
        "Select a variable to view metrics:", metadata["variable"].tolist()
    )
    data_dict_series = metadata[metadata["variable"] == selected_var]
    feature_type = data_dict_series["feature_type"].values[0]

    match feature_type:
        case FeatureType.IDENTIFIER:
            IdentifierCard(df[selected_var], data_dict_series).render()
        case FeatureType.DISCRETE | FeatureType.CONTINUOUS:
            NumericalCard(df[selected_var], data_dict_series).render()
        case FeatureType.NOMINAL | FeatureType.ORDINAL:
            CategoricalCard(df[selected_var], data_dict_series).render()
        case FeatureType.BOOLEAN:
            CategoricalCard(df[selected_var], data_dict_series).render()
        case _:
            st.warning(f"Feature type '{feature_type}' is not supported.")


# ------------------------------------------------------------------------------
# Relationship with Target Variable
# ------------------------------------------------------------------------------
st.subheader("Relationship with Target Variable")

TargetCorrCard(df, metadata, target_var="readmitted").render()

# ------------------------------------------------------------------------------
# Bivariate / Multivariate Analysis
# ------------------------------------------------------------------------------
st.subheader("Bivariate / Multivariate Analysis")

tab1, tab2, tab3 = st.tabs(
    [
        "Numerical <-> Numerical",
        "Categorical <-> Categorical",
        "Numerical <-> Categorical",
    ]
)

with tab1:
    NumericalCorrelationCard(df, metadata).render()
with tab2:
    CategoricalCorrelationCard(df, metadata).render()
with tab3:
    NumericalCategoricalCorrCard(df, metadata).render()

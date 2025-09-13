import re

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go

from constants.feature_type import FeatureType
from constants.color import Color

# TODO: remove the unnecessary df copies - slows down the app


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
        st.markdown("Key findings from the analysis:")

        readmitted_order = ["<30", ">30", "NO"]
        readmitted_color = [Color.DANGER, Color.WARNING, Color.SUCCESS]

        # with st.expander("Age Vs Readmission"):
        with st.container(border=True):
            col1, col2 = st.columns([2, 5])
            with col1:
                st.subheader("Older Patients Linked to Higher Readmission Rates")
                st.markdown(
                    """
                    - Those aged 70 and above particularly prone to short-term (<30 days) readmissions.  
                    - This is expected due to the higher risk of health complications in older adults.  
                """
                )
            with col2:
                # Stacked bar chart showing % of readmission by age group
                values = (
                    pd.crosstab(
                        self.df["age"], self.df["readmitted"], normalize="index"
                    )
                    .reset_index()
                    .melt(id_vars="age", var_name="readmitted", value_name="Proportion")
                )
                values["text"] = values["Proportion"].apply(lambda x: f"{x:.1%}")

                fig = px.bar(
                    values,
                    title="Proportion of Readmission by Age Group",
                    x="age",
                    y="Proportion",
                    color="readmitted",
                    text="text",
                    barmode="stack",
                    category_orders={"readmitted": readmitted_order},
                    color_discrete_sequence=readmitted_color,
                )
                fig.update_traces(
                    marker=dict(line=dict(color=Color.BORDER, width=1)),
                    textposition="auto",
                    textfont_size=16,
                    textfont_color=Color.TEXT,
                )
                fig.update_layout(margin=dict(t=50, b=50, l=0, r=0))
                with st.container(
                    horizontal=True,
                    horizontal_alignment="center",
                    vertical_alignment="center",
                    height="stretch",
                ):
                    st.plotly_chart(fig, use_container_width=True)

        # with st.expander("Time in Hospital Vs Readmission"):
        with st.container(border=True):
            col1, col2 = st.columns([2, 5])
            with col1:
                st.subheader(
                    "Longer Hospital Stays Associated with Higher Readmission Rates"
                )
                st.markdown(
                    """
                    - Patients with extended hospital stays appears to be correlated with higher readmission rates.  
                    - This ***suggests*** that prolonged hospitalization may indicate more severe or complex health issues, leading to increased risk of readmission.  
                    """
                )
            with col2:
                mean_hosp = (
                    self.df.groupby("readmitted")["time_in_hospital"]
                    .mean()
                    .reset_index()
                )
                fig = px.bar(
                    mean_hosp,
                    title="Average Time in Hospital by Readmission Status",
                    x="readmitted",
                    y="time_in_hospital",
                    color="readmitted",
                    category_orders={"readmitted": readmitted_order},
                    color_discrete_sequence=readmitted_color,
                    labels={"time_in_hospital": "Average Time in Hospital (days)"},
                )
                fig.update_traces(
                    marker=dict(line=dict(color=Color.BORDER, width=1)),
                    textposition="auto",
                    texttemplate="%{y:.2f}",
                    textfont_color=Color.TEXT,
                    textfont_size=16,
                )
                fig.update_layout(margin=dict(t=50, b=50, l=0, r=0), showlegend=False)
                with st.container(
                    horizontal=True,
                    horizontal_alignment="center",
                    vertical_alignment="center",
                    height="stretch",
                ):
                    st.plotly_chart(
                        fig, use_container_width=True, use_container_height=True
                    )

        # with st.expander("Medications Vs Readmission"):
        with st.container(border=True):
            col1, col2 = st.columns([2, 5])
            with col1:
                st.subheader("Medication Usage and Readmission Rates")
                st.markdown(
                    """
                    - Patients on a higher number of medications tend to have increased readmission rates.  
                    - Those on diabetic medications are associated with higher readmission rates.
                    - Frequent adjustments to diabetes medications are also linked to higher readmission rates.  
                    This suggests unstable glycemic control and difficulty in stabilizing treatment regimens.  
                    """
                )
            with col2:
                pass

        # with st.expander("Race Vs Readmission"):
        with st.container(border=True):
            col1, col2 = st.columns([2, 5])
            with col1:
                st.subheader("Racial Disparities in Readmission Rates")
                st.markdown(
                    """
                    - African American and Caucasian patients show higher readmission rates.  
                    - They also constitute the majority of the patient population in the dataset (93.7%), driving the overall readmission trends.  
                    - Missing values in this feature weaken the confidence of the finding, but it nonetheless raises questions 
                    about the potential diet / healthcare support in these countries that may affect the readmission rates.
                    """
                )
            with col2:
                with st.container(height="stretch", vertical_alignment="top"):
                    tab1, tab2 = st.tabs(["Readmission Rates", "Racial Distribution"])
                with tab1:
                    race_df = self.df.copy()
                    race_df["readmitted"] = race_df["readmitted"].replace(
                        {"<30": "YES", ">30": "YES"}
                    )

                    values = (
                        pd.crosstab(
                            index=race_df["race"],
                            columns=race_df["readmitted"],
                            normalize="index",
                        )
                        .reset_index()
                        .rename(columns={0: "percentage"})
                    )
                    values["text"] = values["YES"].apply(lambda x: f"{x:.1%}")
                    values = values.sort_values(by="YES", ascending=False)
                    fig = px.bar(
                        x=values["race"],
                        y=values["YES"],
                        title="Readmission Rates for Different Racial Groups",
                        text=values["text"],
                        range_y=[0, 1],
                        labels={"x": "Race", "y": "Readmission Rate"},
                    )
                    fig.update_traces(
                        marker=dict(
                            color=Color.PRIMARY, line=dict(color=Color.BORDER, width=1)
                        ),
                        textposition="auto",
                        textfont_size=14,
                    )
                    fig.update_layout(margin=dict(t=50, b=50, l=0, r=0))

                    # horizontal line for average readmission rate
                    avg_readmit_rate = (self.df["readmitted"] != "NO").mean()
                    fig.add_hline(
                        y=avg_readmit_rate,
                        line_dash="dash",
                        line_color=Color.TEXT,
                        annotation_text=f"Average Readmission Rate: {avg_readmit_rate:.1%}",
                        annotation_position="top right",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                with tab2:
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
                    st.plotly_chart(fig, use_container_width=True, key="race_target")

        # with st.expander("Inpatient Visits Vs Readmission"):
        with st.container(border=True):
            col1, col2 = st.columns([2, 5])
            with col1:
                st.subheader("Frequent Inpatient Visits and Readmission Rates")
                st.markdown(
                    """
                    - Patients with a history of frequent inpatient visits are associated with higher readmission rates.  
                    - This trend is particularly pronounced for short-term (<30 days) readmissions.  
                    - This suggest repeated hospitalizations may indicate chronic or poorly managed health conditions.  
                    """
                )
            with col2:

                mean_proc = (
                    self.df.groupby("readmitted")["number_inpatient"]
                    .mean()
                    .reset_index()
                )

                fig = px.bar(
                    mean_proc,
                    title="Average Number of Inpatient Visits by Readmission Status",
                    x="readmitted",
                    y="number_inpatient",
                    color="readmitted",
                    category_orders={"readmitted": readmitted_order},
                    color_discrete_sequence=readmitted_color,
                    labels={"number_inpatient": "Average Number of Inpatient Visits"},
                )
                fig.update_traces(
                    marker=dict(line=dict(color=Color.BORDER, width=1)),
                    textposition="auto",
                    texttemplate="%{y:.2f}",
                    textfont_color=Color.TEXT,
                    textfont_size=16,
                )
                fig.update_layout(margin=dict(t=50, b=50, l=0, r=0))
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        # with st.expander("Discharge Disposition Vs Readmission"):
        with st.container(border=True):
            col1, col2 = st.columns([2, 5])
            with col1:
                st.subheader(
                    "Patients Discharged to Home Linked to Lower Readmission Rates"
                )
                st.markdown(
                    """
                    - Patients discharged to home are associated with lower rates of readmission compared to those discharged to other facilities.
                    - This suggests that patients requiring further care or rehabilitation may have more complex health needs, leading to higher readmission rates.  
                    """
                )
            with col2:
                home_df = self.df.copy()
                mapping = {x: "Other" for x in range(2, 29)}
                mapping[1] = "Home"
                home_df["discharge_disposition_id"] = home_df[
                    "discharge_disposition_id"
                ].replace(mapping)
                home_df["readmitted"] = home_df["readmitted"].replace(
                    {"<30": "YES", ">30": "YES"}
                )

                values = (
                    pd.crosstab(
                        index=home_df["discharge_disposition_id"],
                        columns=home_df["readmitted"],
                        normalize="index",
                    )
                    .reset_index()
                    .rename(columns={0: "percentage"})
                )
                values["text"] = values["YES"].apply(lambda x: f"{x:.1%}")
                values = values.sort_values(by="YES", ascending=True)
                fig = px.bar(
                    x=values["discharge_disposition_id"],
                    y=values["YES"],
                    title="Readmission Rates by Discharge Disposition",
                    text=values["text"],
                    range_y=[0, 1],
                    labels={"x": "Discharge Disposition", "y": "Readmission Rate"},
                )
                fig.update_traces(
                    marker=dict(
                        color=Color.PRIMARY, line=dict(color=Color.BORDER, width=1)
                    ),
                    textposition="auto",
                    textfont_size=16,
                )
                fig.update_layout(margin=dict(t=50, b=50, l=0, r=0))
                fig.add_annotation(
                    x=0.05,
                    y=0.95,
                    xref="paper",
                    yref="paper",
                    text="Patients discharged to home have lower readmission rates."
                    "<br>Those discharged to other facilities may have more complex health needs.",
                    showarrow=False,
                    font=dict(color=Color.ANNOTATION_TEXT, size=14),
                    bgcolor=Color.ANNOTATION_BG,
                    bordercolor=Color.BORDER,
                    borderpad=5,
                )

                avg_readmit_rate = (self.df["readmitted"] != "NO").mean()
                fig.add_hline(
                    y=avg_readmit_rate,
                    line_dash="dash",
                    line_color=Color.TEXT,
                    annotation_text=f"Average Readmission Rate: {avg_readmit_rate:.1%}",
                )

                st.plotly_chart(fig, use_container_width=True, key="discharge_target")

        # with st.expander("Emergency Visits Vs Readmission"):
        with st.container(border=True):
            col1, col2 = st.columns([2, 5])
            with col1:
                st.subheader(
                    "Frequent Emergency Visits Linked to Higher Readmission Rates"
                )
                st.markdown(
                    """
                    - Higher number of emergency visits correlates with increased readmission rates.  
                    - This sugggests inadequate disease control or limited access to regular outpatient care.  
                    """
                )
            with col2:
                tab1, tab2 = st.tabs(
                    ["Average Emergency Visits", "Emergency Visits Distribution"]
                )
                with tab1:
                    # vertical bar chart showing mean number of emergency visits by readmission status
                    mean_emerg = (
                        self.df.groupby("readmitted")["number_emergency"]
                        .mean()
                        .reset_index()
                    )
                    fig = px.bar(
                        mean_emerg,
                        title="Average Number of Emergency Visits by Readmission Status",
                        x="readmitted",
                        y="number_emergency",
                        color="readmitted",
                        category_orders={"readmitted": readmitted_order},
                        color_discrete_sequence=readmitted_color,
                        labels={
                            "number_emergency": "Average Number of Emergency Visits"
                        },
                    )
                    fig.update_traces(
                        marker=dict(line=dict(color=Color.BORDER, width=1)),
                        textposition="auto",
                        texttemplate="%{y:.2f}",
                        textfont_color=Color.TEXT,
                        textfont_size=16,
                    )
                    fig.update_layout(
                        margin=dict(t=50, b=50, l=0, r=0), showlegend=False
                    )

                    avg_emerg = self.df["number_emergency"].mean()
                    fig.add_hline(
                        y=avg_emerg,
                        line_dash="dash",
                        line_color=Color.TEXT,
                        annotation_text=f"Average Emergency Visits: {avg_emerg:.2f}",
                        annotation_position="top right",
                    )

                    with st.container(
                        horizontal=True,
                        horizontal_alignment="center",
                        vertical_alignment="center",
                        height="stretch",
                    ):
                        st.plotly_chart(
                            fig, use_container_width=True, use_container_height=True
                        )
            with tab2:
                emerg_df = self.df.copy()

                # bin to 0, [1-2], [3-5], [6-10], [11-20], >20
                bins = [-1, 0, 2, 5, 10, 20, emerg_df["number_emergency"].max()]
                labels = ["0", "[1-2]", "[3-5]", "[6-10]", "[11-20]", ">20"]
                emerg_df["emerg_bin"] = pd.cut(
                    emerg_df["number_emergency"], bins=bins, labels=labels
                )

                bin_counts = (
                    emerg_df["emerg_bin"].value_counts().sort_index().reset_index()
                )
                bin_counts.columns = ["emerg_bin", "count"]
                total = bin_counts["count"].sum()
                bin_counts["percent"] = bin_counts["count"] / total

                # bar chart of emergency visits distribution
                fig = px.bar(
                    emerg_df["emerg_bin"].value_counts().sort_index().reset_index(),
                    x="emerg_bin",
                    y="count",
                    title="Distribution of Emergency Visits",
                    labels={
                        "count": "Number of Patient Encounters",
                        "emerg_bin": "Number of Emergency Visits (binned)",
                    },
                )
                fig.update_traces(
                    marker=dict(
                        color=Color.PRIMARY, line=dict(color=Color.BORDER, width=1)
                    ),
                    text=[
                        f"{c:,} ({p:.1%})"
                        for c, p in zip(bin_counts["count"], bin_counts["percent"])
                    ],
                    textposition="auto",
                    textfont_color=Color.TEXT,
                    textfont_size=16,
                )
                fig.add_annotation(
                    x=0.2,
                    y=0.95,
                    xref="paper",
                    yref="paper",
                    text=f"The distribution is heavily right-skewed with a <b>mean of {emerg_df['number_emergency'].mean():.2f}</b> visits."
                    "<br><b>88.8%</b> of encounters had no emergency visits."
                    "<br>Only <b>0.4%</b> of encounters had more than 5 emergency visits.",
                    showarrow=False,
                    font=dict(color=Color.ANNOTATION_TEXT, size=14),
                    bgcolor=Color.ANNOTATION_BG,
                    bordercolor=Color.BORDER,
                    borderpad=5,
                )
                fig.update_layout(margin=dict(t=50, b=50, l=0, r=0))
                st.plotly_chart(fig, use_container_width=True, key="emergency_target")

        # with st.expander("Primary Diagnosis Vs Readmission"):
        with st.container(border=True):
            col1, col2 = st.columns([2, 5])
            with col1:
                st.subheader("Primary Diagnosis Category and Readmission Rates")
                st.markdown(
                    """
                    - Patients admitted for certain mediacal conditions are associated with higher readmission rates.  
                    - This underscores the impact of comorbidities on readmission risk.  
                    """
                )
            with col2:
                pri_diag_df = self.df.copy()

                def mapping(icd_9: str) -> str:

                    if icd_9.startswith("V") or icd_9.startswith("E"):
                        return "External Causes"

                    # extract the numeric part before the decimal point
                    icd_9_main = re.match(r"^(\d{1,3})", icd_9)
                    if not icd_9_main:
                        return "Unknown"

                    match int(icd_9_main.group()):
                        case x if 1 <= x <= 139:
                            return "Infectious"
                        case x if 140 <= x <= 239:
                            return "Neoplasms"
                        case x if 240 <= x <= 279:
                            return "Endocrine"
                        case x if 280 <= x <= 289:
                            return "Blood"
                        case x if 290 <= x <= 319:
                            return "Mental"
                        case x if 320 <= x <= 389:
                            return "Nervous"
                        case x if 390 <= x <= 459:
                            return "Circulatory"
                        case x if 460 <= x <= 519:
                            return "Respiratory"
                        case x if 520 <= x <= 579:
                            return "Digestive"
                        case x if 580 <= x <= 629:
                            return "Genitourinary"
                        case x if 630 <= x <= 679:
                            return "Pregnancy"
                        case x if 680 <= x <= 709:
                            return "Skin"
                        case x if 710 <= x <= 739:
                            return "Musculoskeletal"
                        case x if 740 <= x <= 759:
                            return "Congenital"
                        case x if 760 <= x <= 779:
                            return "Perinatal"
                        case x if 780 <= x <= 799:
                            return "Symptoms"
                        case x if 800 <= x <= 999:
                            return "Injury"
                        case _:
                            return "Other"

                pri_diag_df["diag_1"] = pri_diag_df["diag_1"].apply(mapping)
                pri_diag_df["readmitted"] = pri_diag_df["readmitted"].replace(
                    {"<30": "YES", ">30": "YES"}
                )

                values = (
                    pd.crosstab(
                        index=pri_diag_df["diag_1"],
                        columns=pri_diag_df["readmitted"],
                        normalize="index",
                    )
                    .reset_index()
                    .rename(columns={0: "percentage"})
                )
                values["text"] = values["YES"].apply(lambda x: f"{x:.1%}")
                values = values.sort_values(by="YES", ascending=True)
                fig = px.bar(
                    values,
                    x="diag_1",
                    y="YES",
                    title="Readmission Rates by Primary Diagnosis Category",
                    text=values["text"],
                    range_y=[0, 1],
                    labels={
                        "diag_1": "Primary Diagnosis Category",
                        "YES": "Readmission Rate",
                    },
                )
                fig.update_traces(
                    marker=dict(
                        color=Color.PRIMARY, line=dict(color=Color.BORDER, width=1)
                    ),
                    textposition="auto",
                    textfont_size=16,
                )
                avg_readmit_rate = (self.df["readmitted"] != "NO").mean()
                fig.add_hline(
                    y=avg_readmit_rate,
                    line_dash="dash",
                    line_color=Color.TEXT,
                    annotation_position="top left",
                    annotation_text=f"Average Readmission Rate: {avg_readmit_rate:.1%}",
                )
                fig.add_annotation(
                    x=0.05,
                    y=0.95,
                    xref="paper",
                    yref="paper",
                    text="Certain primary diagnosis categories "
                    "<br>are linked to higher readmission rates.",
                    showarrow=False,
                    font=dict(color=Color.ANNOTATION_TEXT, size=14),
                    bgcolor=Color.ANNOTATION_BG,
                    bordercolor=Color.BORDER,
                    borderpad=5,
                )
                fig.update_layout(showlegend=False, margin=dict(t=50, b=50, l=0, r=0))
                with st.container(
                    horizontal=True,
                    horizontal_alignment="center",
                    vertical_alignment="center",
                    height="stretch",
                ):
                    st.plotly_chart(fig, use_container_width=True)

        # with st.expander("HbA1c Level Vs Readmission"):
        with st.container(border=True):
            col1, col2 = st.columns([2, 5])
            with col1:
                st.subheader(
                    "Higher HbA1c Levels Associated with Increased Readmission Rates"
                )
                st.markdown(
                    """
                    - Patients with elevated HbA1c levels exhibit higher readmission rates.  
                    - This is consistent with the understanding that poor glycemic control increases the risk of complications and hospitalizations.  
                    """
                )
            with col2:
                hba1c_df = self.df.copy()
                hba1c_df["readmitted"] = hba1c_df["readmitted"].replace(
                    {"<30": "YES", ">30": "YES"}
                )
                hba1c_order = [">8", ">7", "Norm"]
                hba1c_color = [Color.DANGER, Color.WARNING, Color.SUCCESS]
                values = (
                    pd.crosstab(
                        index=hba1c_df["A1Cresult"],
                        columns=hba1c_df["readmitted"],
                        normalize="index",
                    )
                    .reset_index()
                    .rename(columns={0: "percentage"})
                )
                values["text"] = values["YES"].apply(lambda x: f"{x:.1%}")
                values = values.sort_values(by="YES", ascending=True)
                fig = px.bar(
                    values,
                    x="A1Cresult",
                    y="YES",
                    title="Readmission Rates by HbA1c Levels",
                    text=values["text"],
                    range_y=[0, 1],
                    labels={"A1Cresult": "HbA1c Levels", "YES": "Readmission Rate"},
                    color="A1Cresult",
                    category_orders={"A1Cresult": hba1c_order},
                    color_discrete_sequence=hba1c_color,
                )
                fig.update_traces(
                    marker=dict(line=dict(color=Color.BORDER, width=1)),
                    textposition="auto",
                    textfont_color=Color.TEXT,
                    textfont_size=16,
                )
                fig.update_layout(margin=dict(t=50, b=50, l=0, r=0))
                fig.add_annotation(
                    x=0.05,
                    y=0.95,
                    xref="paper",
                    yref="paper",
                    text="Patients with poor glycemic control associated "
                    "<br>with higher readmission rates.",
                    showarrow=False,
                    font=dict(color=Color.ANNOTATION_TEXT, size=14),
                    bgcolor=Color.ANNOTATION_BG,
                    bordercolor=Color.BORDER,
                    borderpad=5,
                )
                fig.update_layout(showlegend=False, margin=dict(t=50, b=50, l=0, r=0))
                with st.container(
                    horizontal=True,
                    horizontal_alignment="center",
                    vertical_alignment="center",
                    height="stretch",
                ):
                    st.plotly_chart(fig, use_container_width=True)

        # with st.expander("Number of Procedures Vs Readmission"):
        with st.container(border=True):
            col1, col2 = st.columns([2, 5])
            with col1:
                st.subheader(
                    "More Procedures During Stay Associated with Higher Readmission Rates"
                )
                st.markdown(
                    """
                    - Patients undergoing multiple procedures during their hospital stay tend to have higher readmission rates.  
                    - This suggests that more complex medical interventions may indicate severe health conditions, leading to increased risk of readmission.  
                    """
                )
            with col2:

                mean_proc = (
                    self.df.groupby("readmitted")["num_procedures"].mean().reset_index()
                )

                fig = px.bar(
                    mean_proc,
                    title="Average Number of Procedures by Readmission Status",
                    x="readmitted",
                    y="num_procedures",
                    color="readmitted",
                    category_orders={"readmitted": readmitted_order},
                    color_discrete_sequence=readmitted_color,
                    labels={"num_procedures": "Average Number of Procedures"},
                )
                fig.update_traces(
                    marker=dict(line=dict(color=Color.BORDER, width=1)),
                    textposition="auto",
                    texttemplate="%{y:.2f}",
                    textfont_color=Color.TEXT,
                    textfont_size=16,
                )
                fig.update_layout(showlegend=False, margin=dict(t=50, b=50, l=0, r=0))
                st.plotly_chart(fig, use_container_width=True)

    def _plot_boxplot(self):
        readmitted_order = ["<30", ">30", "NO"]
        readmitted_color = [Color.DANGER, Color.WARNING, Color.SUCCESS]
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

        with st.container(border=True):
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

        # exclude target variable and medications
        categorical_cols = categorical_cols[categorical_cols != self.target_var]
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
                index=0,  # defult show 1
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

        with st.container(border=True):
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
        self._render_summary_description()
        with st.expander("Interactive Analysis (Target Variable)", expanded=False):
            tab1, tab2 = st.tabs(
                [
                    "Target <-> Numerical",
                    "Target <-> Categorical (Excl Medications)",
                ]
            )

            with tab1:
                self._plot_boxplot()
            with tab2:
                self._plot_barchart()

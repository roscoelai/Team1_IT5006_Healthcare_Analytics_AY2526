#!/usr/bin/env python
# app.py
# 2025-08-24
# Roscoe

"""
TODO:
- [ ] Read up on caching, might get laggy with too many plots
"""

import altair as alt
import polars as pl
import polars.selectors as cs
import streamlit as st


df = pl.read_parquet("data/diabetic_data.parquet")
for s in df:
    if s.is_null().any():
        df = df.with_columns(pl.col(s.name).fill_null("None"))


st.title("Diabetes dataset dashboard")

nrow, ncol = df.shape
col1, col2 = st.columns(2)
with col1:
    st.metric("Number of rows:", nrow)
with col2:
    st.metric("Number of columns:", ncol)

dtypes = {}
for s in df:
    dtype = str(type(s.dtype))
    dtypes[dtype] = dtypes.get(dtype, []) + [s.name]
# st.write(dtypes)


# st.write("## Raw dataframe")
# 
# st.dataframe(df)


st.write("---")
st.write("## Bar chart for enums")

selections = st.multiselect("Select enum variable:", dtypes["Enum"])
if selections:
    for selection in selections:
        vc = df[selection].value_counts().sort(selection)
        vc = vc.drop_nulls()
        chart = alt.Chart(vc).mark_bar().encode(
            x=alt.X(f"{selection}:O", sort=vc[selection].to_list()),
            y="count:Q"
        ).properties(
            title=selection
        )
        st.altair_chart(chart, use_container_width=True)


st.write("---")
st.write("## Hist for numeric")
st.write("All numeric columns are integers, scatterplots might not be pretty.")
st.write("Decide after looking at histograms.")

selections = st.multiselect("Select numeric variable:", dtypes["Int64"])
descs = df[dtypes["Int64"]].describe()
# st.write(descs)
if selections:
    for selection in selections:
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X(f"{selection}:Q", bin=True),
            y="count()",
        ).properties(
            title=selection
        )
        st.altair_chart(chart, use_container_width=True)
        d2 = pl.DataFrame(dict(zip(descs["statistic"], descs[selection])))
        st.dataframe(d2, use_container_width=True)
        # st.dataframe(d2)



st.write("---")
st.write("## Special investigations")

st.write("### `patient_nbr`s are not unique")

s = df["patient_nbr"]
vc = s.filter(s.is_duplicated()).value_counts()
encounters = vc["count"].rename("encounters").value_counts(sort=True)
# st.write(encounters)
st.bar_chart(encounters, x="encounters", y="count")




# All numeric columns are integers, scatter_charts might not be a good idea.
# st.scatter_chart(df, x="num_procedures", y="time_in_hospital")

# TODO: Count number of occurrences of patient_nbr



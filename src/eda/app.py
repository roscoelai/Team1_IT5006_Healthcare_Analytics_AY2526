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


@st.cache_data
def read_data(source: str="data/diabetic_data.parquet") -> pl.DataFrame:
    return pl.read_parquet(source)


@st.cache_data
def collect_dtypes(df: pl.DataFrame) -> dict[str, str]:
    dtypes = {}
    for s in df:
        dtype = str(type(s.dtype))
        dtypes[dtype] = dtypes.get(dtype, []) + [s.name]
    return dtypes


@st.cache_data
def get_num_descs(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(cs.numeric()).describe()


# ---

df = read_data()


st.title("Diabetes dataset dashboard")

nrow, ncol = df.shape
col1, col2 = st.columns(2)
with col1:
    st.metric("Number of rows:", nrow)
with col2:
    st.metric("Number of columns:", ncol)

dtypes = collect_dtypes(df)
# st.write(dtypes)


# st.write("## Raw dataframe")
# 
# st.dataframe(df)


st.write("---")
st.write("## Bar chart for enums")
st.write("> Try not to use multiselect, might get rather laggy...")

col1, col2 = st.columns(2)
with col1:
    selection = st.selectbox("Select enum variable:", dtypes["Enum"])
with col2:
    groupby_var = st.selectbox("Select group by variable:", dtypes["Enum"][-1:] + dtypes["Enum"][:-1])
if selection:
    df2 = df.group_by(selection, groupby_var).agg(count=pl.len()).sort(selection, groupby_var).drop_nulls()
    chart = alt.Chart(df2).mark_bar().encode(
        x=alt.X(f"{selection}:O", sort=df2[selection]),
        y="count:Q",
        color=f"{groupby_var}:O",
    ).properties(
        title=selection
    )
    st.altair_chart(chart, use_container_width=True)


st.write("---")
st.write("## Hist for numeric")
st.write("All numeric columns are integers, scatterplots might not be pretty.")
st.write("Decide after looking at histograms.")

# selections = st.multiselect("Select numeric variable:", dtypes["Int64"])
selection = st.selectbox("Select numeric variable:", dtypes["Int64"])
num_descs = get_num_descs(df)
# st.write(num_descs)
if selection:
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X(f"{selection}:Q", bin=True),
        y="count()",
    ).properties(
        title=selection
    )
    st.altair_chart(chart, use_container_width=True)
    d2 = pl.DataFrame(dict(zip(num_descs["statistic"], num_descs[selection])))
    st.dataframe(d2)
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



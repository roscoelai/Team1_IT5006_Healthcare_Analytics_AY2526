"""Main entry point for the streamlit interactive dashboard"""

import os
import streamlit as st
import pandas as pd

BASE_PAGE_DIR = "./pages"


@st.cache_data
def load_csv_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def create_pages() -> list[st.Page]:
    intro = st.Page(
        os.path.join(BASE_PAGE_DIR, "introduction.py"),
        title="Introduction",
        icon="🏠",
        url_path="introduction",
    )
    literature_review = st.Page(
        os.path.join(BASE_PAGE_DIR, "literature_review.py"),
        title="Literature Review",
        icon="📚",
        url_path="literature",
    )
    data_prepro = st.Page(
        os.path.join(BASE_PAGE_DIR, "data_preprocessing.py"),
        title="Data Preprocessing",
        icon="🔧",
        url_path="preprocessing",
    )
    eda = st.Page(
        os.path.join(BASE_PAGE_DIR, "exploratory_analysis.py"),
        title="Exploratory Analysis",
        icon="🔍",
        url_path="eda",
    )
    dashboard = st.Page(
        os.path.join(BASE_PAGE_DIR, "dashboard.py"),
        title="Dashboard",
        icon="📊",
        url_path="dashboard",
    )

    return [intro, literature_review, data_prepro, eda, dashboard]


def main():
    pages = create_pages()
    pg = st.navigation(pages)
    st.set_page_config(page_title="Diabetic Data Dashboard", layout="wide")
    pg.run()


if __name__ == "__main__":
    main()

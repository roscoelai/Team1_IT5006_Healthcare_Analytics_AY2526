"""Main entry point for the streamlit interactive dashboard"""

import os
import sys

import streamlit as st
import pandas as pd

sys.path.append(os.path.dirname(__file__))

from utils.dataloader import DataLoader

# Constants
BASE_PAGE_DIR = "./pages"
DATA_PATH = "./data/diabetic_data.csv"
METADATA_PATH = "./data/diabetes_datadict.csv"


def create_pages() -> list[st.Page]:
    """Create the pages for the dashboard

    Returns:
        list[st.Page]: List of pages for the dashboard, in the order they should appear
    """
    intro = st.Page(
        os.path.join(BASE_PAGE_DIR, "introduction.py"),
        title="Introduction",
        icon="🏠",
        url_path="introduction",
    )
    eda = st.Page(
        os.path.join(BASE_PAGE_DIR, "exploratory_analysis.py"),
        title="Exploratory Analysis",
        icon="🔍",
        url_path="eda",
    )

    return [intro, eda]


def main():

    # Initialize DataLoader
    DataLoader.init(DATA_PATH, METADATA_PATH)

    # Create pages
    pages = create_pages()

    # Setup navigation
    pg = st.navigation(pages)

    # Set page config and run
    st.set_page_config(page_title="Diabetic Data Dashboard", layout="wide")
    pg.run()


if __name__ == "__main__":
    main()

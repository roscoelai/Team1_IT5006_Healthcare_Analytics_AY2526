import pandas as pd
import streamlit as st


class DataDictCard:
    """A card that displays the data dictionary of the dataset."""

    def __init__(self, metadata: pd.DataFrame):
        self.metadata = metadata
        self.feat_options = self.metadata["feature_type"].unique().tolist()
        self.cat_options = self.metadata["category"].unique().tolist()

        # Initialize session state for filters if not already set
        st.session_state["feature_type_filter"] = st.session_state.get(
            "feature_type_filter", self.feat_options
        )
        st.session_state["category_filter"] = st.session_state.get(
            "category_filter", self.cat_options
        )

    def render(self):
        """Render the data dictionary card."""
        with st.container(border=True):
            st.info(
                """🔍 Use the filters on the left to explore specific feature types or categories. 
                Click on the column headers to sort by a specific variable."""
            )

            col1, col2 = st.columns([2, 5])

            # TODO: too much duplicate code. refactor later.

            with col1:
                # Render filters UI
                self._render_filters()
            with col2:
                # Render data dictionary table
                self._render_datadict()

    def _render_filters(self):
        """Render the filtering options for the data dictionary."""

        with st.container(border=True, height="stretch"):

            st.subheader("**Filter**")

            # TODO: duplicated code for filtering. Refactor later.

            # Filter by feature type
            with st.expander("By Feature Type"):
                with st.container(horizontal=True, horizontal_alignment="right"):
                    select_all_feat = st.button(
                        label="Select All",
                        key="select_all_feat",
                        type="tertiary",
                        use_container_width=False,
                    )

                    clear_all_feat = st.button(
                        label="Clear All",
                        key="clear_all_feat",
                        type="tertiary",
                        use_container_width=False,
                    )
                    if select_all_feat:
                        st.session_state["feature_type_filter"] = (
                            self.metadata["feature_type"].unique().tolist()
                        )
                        st.rerun()
                    if clear_all_feat:
                        st.session_state["feature_type_filter"] = []
                        st.rerun()

                feature_type_filter = []
                for option in self.feat_options:
                    checked = st.checkbox(
                        option,
                        value=option in st.session_state.get("feature_type_filter", []),
                        key=f"feature_type_{option}",
                    )
                    if checked:
                        feature_type_filter.append(option)
                st.session_state["feature_type_filter"] = feature_type_filter

            # Filter by category
            with st.expander("By Feature Category"):
                with st.container(horizontal=True, horizontal_alignment="right"):
                    select_all_cat = st.button(
                        label="Select All",
                        key="select_all_cat",
                        type="tertiary",
                        use_container_width=False,
                    )
                    clear_all_feature_types = st.button(
                        label="Clear All",
                        key="dclear_all_feature_types",
                        type="tertiary",
                        use_container_width=False,
                    )
                    if select_all_cat:
                        st.session_state["category_filter"] = (
                            self.metadata["category"].unique().tolist()
                        )
                        st.rerun()
                    if clear_all_feature_types:
                        st.session_state["category_filter"] = []
                        st.rerun()

                category_filter = []
                for option in self.cat_options:
                    checked = st.checkbox(
                        option,
                        value=option in st.session_state.get("category_filter", []),
                        key=f"category_{option}",
                    )
                    if checked:
                        category_filter.append(option)
                st.session_state["category_filter"] = category_filter

    def _render_datadict(self):
        """Render the data dictionary table based on the selected filters."""

        # Apply filters to the metadata
        datadict = self.metadata[
            self.metadata["feature_type"].isin(
                st.session_state.get("feature_type_filter", [])
            )
            & self.metadata["category"].isin(
                st.session_state.get("category_filter", [])
            )
        ]
        datadict.reset_index(inplace=True, drop=True)

        # Start index from 1 instead of 0 - for better readability
        datadict.index += 1

        # Remove unnecessary columns for display
        datadict = datadict.drop(columns=["value_counts", "data_type"])

        # Indicator of number of variables left after filtering
        filtered = len(st.session_state.get("feature_type_filter", [])) != len(
            self.feat_options
        ) or len(st.session_state.get("category_filter", [])) != len(self.cat_options)
        st.write(
            f'<i>Total variables: {datadict.shape[0]} {"(filtered)" if filtered else ""}</i>',
            unsafe_allow_html=True,
        )

        # Display the data dictionary table
        st.write(datadict)

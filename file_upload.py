import streamlit as st
import pandas as pd
import io

def show():
    st.title('Model trainer')
    st.header("File upload")
    st.caption(
        'A machine learning tool that enables you make preprocessing '
        'in your own dataset from specifying the columns you want to feed the model'
        ' to drop duplicated, encode categorical data, scaling values..., and much more,'
        ' even you can see the relationship between data attributes and finally you can choos'
        ' the model that scores the best accurecy'
    )

    uploaded_file = st.file_uploader(
        "Choose a dataset file (CSV, Excel, or JSON)", 
        type=["csv", "xlsx", "xls", "json"]
    )

    # Only update session_state.file if a new file is uploaded
    if uploaded_file is not None:
        st.session_state.file = uploaded_file
        st.session_state.df = None  # Reset df so it reloads

    # Use the file from session_state if available
    file = st.session_state.get('file', None)
    df = st.session_state.get('df', None)

    if file is not None and df is None:
        file.seek(0)  # Reset pointer to start
        file_name = file.name.lower()
        try:
            if file_name.endswith('.csv'):
                df = pd.read_csv(file)
            elif file_name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file)
            elif file_name.endswith('.json'):
                try:
                    df = pd.read_json(file)
                except ValueError:
                    df = None
            else:
                st.error("Unsupported file type.")
                df = None

            if df is not None:
                st.session_state['df'] = df
        except Exception as e:
            st.error(f"Error loading file: {e}")

    df = st.session_state.get('df', None)
    if df is not None:
        st.success("Data set uploaded successfuly")
        st.write("Preview of uploaded dataset:")
        st.dataframe(df.head())
        col1, col2 = st.columns([1, 5])
        with col1:
            show_info = st.button("information")
        with col2:
            st.write("view general information about data")

        if show_info:
            buffer = io.StringIO()
            df.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)

        st.markdown("---")  

        col3, col4 = st.columns([1, 5])
        with col3:
            show_stats = st.button("describtion")
        with col4:
            st.write("View meta statistics")

        if show_stats:
            st.write(df.describe())

def get_df():
    return st.session_state.get('df', None)
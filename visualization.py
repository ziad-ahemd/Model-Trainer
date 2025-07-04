import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from file_upload import *

def show():
    st.title('Data visualization')

    if 'df' not in st.session_state:
        st.session_state.df = get_df()
    df = st.session_state.df
    if df.empty:
        st.warning("Dataset is empty, please upload a proper dataset")
        return
# Initialize numeric and non_numeric columns if not present
    if df is not None:
        st.session_state.numeric = [col for col in df.columns.values if pd.api.types.is_numeric_dtype(df[col])]
        st.session_state.non_numeric = [col for col in df.columns.values if not pd.api.types.is_numeric_dtype(df[col])]
    else:
        st.warning("No dataset loaded. Please upload a dataset first.")
        return
    numeric_data = st.session_state.numeric
    categorical_data = st.session_state.non_numeric
    st.dataframe(df.head())

    def choose_one_column(columns):
        x = st.selectbox('Choose a column to visualize', columns)
        if x:
            return x
        else:
            st.error('Please enter columns needed to visualize')
            return False
    def choose_two_columns(columns):
        x = st.selectbox('Choose a column to visualize', columns, key='x')
        y = st.selectbox('Choose a column to visualize', columns, key='y')
        if x and y:
            return x, y
        else:
            st.error('Please enter columns needed to visualize')
            return False, False

    plot_method = st.selectbox('Choose a plot', ['Line plot', 'Bar plot', 'Histogram plot', 'Box plot',
     'Scatter plot', 'Heatmap plot', 'Pie plot', 'Count plot'])
    if plot_method == 'Line plot':
        x, y = choose_two_columns(numeric_data)
        plot_btn = st.button('Plot')
        if plot_btn and x:
            fig, ax = plt.subplots()
            plt.title(f'{x} VS {y}')
            ax.plot(df[x], df[y])
            plt.xlabel(x)
            plt.ylabel(y)
            st.pyplot(fig)
    elif plot_method == 'Bar plot':
        x, y = choose_two_columns(df.columns)
        plot_btn = st.button('Plot')
        if plot_btn and x:
            fig, ax = plt.subplots()
            plt.title(f'{x} VS {y}')
            ax.bar(df[x], df[y])
            plt.xlabel(x)
            plt.ylabel(y)
            st.pyplot(fig)
    elif plot_method == 'Histogram plot':
        x = choose_one_column(numeric_data)
        plot_btn = st.button('Plot')
        if plot_btn and x:
            fig, ax = plt.subplots()
            ax.hist(df[x])
            plt.xlabel(x)
            st.pyplot(fig)
    elif plot_method == 'Box plot':
        x = choose_one_column(numeric_data)
        plot_btn = st.button('Plot')
        if plot_btn and x:
            fig, ax = plt.subplots()
            ax.boxplot(df[x])
            plt.ylabel(x)
            st.pyplot(fig)
    elif plot_method == 'Scatter plot':
        x, y = choose_two_columns(numeric_data)
        plot_btn = st.button('Plot')
        if plot_btn and x:
            fig, ax = plt.subplots()
            plt.title(f'{x} VS {y}')
            ax.scatter(df[x], df[y])
            plt.xlabel(x)
            plt.ylabel(y)
            st.pyplot(fig)
    elif plot_method == 'Heatmap plot':
        plot_btn = st.button('Plot')
        if plot_btn:
            fig, ax = plt.subplots()
            sns.heatmap(df[numeric_data].corr(), annot=True, square=True, ax=ax)
            st.pyplot(fig)
    elif plot_method == 'Count plot':
        x = choose_one_column(categorical_data)
        plot_btn = st.button('Plot')
        if plot_btn and x:
            fig, ax = plt.subplots()
            sns.countplot(data=df, x=x, ax=ax)
            plt.xlabel(x)
            plt.ylabel(f'Counts of {x}')
            st.pyplot(fig)
    elif plot_method == 'Pie plot':
        labels = choose_one_column(categorical_data)
        plot_btn = st.button('Plot')
        if plot_btn and labels:
            counts = df[labels].value_counts()
            fig, ax = plt.subplots()
            plt.pie(x=counts, labels=df[labels].unique(), autopct='%1.1f%%')
            st.pyplot(fig)

    else:
        st.error('Please select a plot')
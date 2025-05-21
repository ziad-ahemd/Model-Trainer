import streamlit as st
import numpy as np
import pandas as pd
from file_upload import *
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


def show():
    st.title('Data preprocessing')

    # Retreving dataframe
    if 'df' not in st.session_state:
        st.session_state.df = get_df()
    df = st.session_state.df

    # Initialize numeric and non_numeric columns if not present
    if df is not None:
<<<<<<< HEAD
        st.session_state.numeric = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        st.session_state.non_numeric = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
=======
        st.session_state.numeric = [col for col in df.columns.values if pd.api.types.is_numeric_dtype(df[col])]
        st.session_state.non_numeric = [col for col in df.columns.values if not pd.api.types.is_numeric_dtype(df[col])]
>>>>>>> a7aed5ab2ddd9770497f1227eb9b52b4f9bfece1
    else:
        st.warning("No dataset loaded. Please upload a dataset first.")
        return
    numeric_data = st.session_state.numeric
    categorical_data = st.session_state.non_numeric
    st.dataframe(df)

<<<<<<< HEAD
=======
    st.markdown('---')

>>>>>>> a7aed5ab2ddd9770497f1227eb9b52b4f9bfece1
    # Drop column
    st.header('Features filtering')
    col_to_drop = st.selectbox(label='Choose irrelavint column to drop',options=df.columns)
    drop_column_btn = st.button('Drop column')
    if drop_column_btn:
        if col_to_drop:
            df = df.drop(col_to_drop, axis=1)
            st.session_state.df = df
            st.dataframe(df)
            st.success(f'{col_to_drop}, droped successfuly')
        else:
            st.error('Please select a valid column')
    st.markdown('---')

    # Handling duplicates
    st.header('Duplicates')
    st.subheader('Detect duplicates')
    show_duplicate_btn = st.button('Show duplicates')
    if show_duplicate_btn:
        st.write(df[df.duplicated()])
    st.subheader('Remove duplicates')
    drop_duplicate_btn = st.button('Drop duplicates')
    if drop_duplicate_btn:
        df = df.drop_duplicates()
        st.session_state.df = df
        st.write(df.duplicated().sum())
        st.success('Duplicated removed successfuly')


    st.markdown('---')

    # Handling missing values
    st.header('Missing values')
    st.subheader('Detect missing values')
    show_nulls_btn = st.button('Show missing values')
    if show_nulls_btn:
        st.write(df.isna().sum())
    st.subheader('Deal with missing values')
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write('Drop missing values')
        drop_nulls_btn = st.button('Drop missing values')
        if drop_nulls_btn:
            df = df.dropna()
            st.session_state.df = df
            st.write(df.isna().sum())
            st.success('Missing values droped successfuly')
    with col2:
        st.write('Impute missing values')
        col_to_impute = st.multiselect('Choose one or more column to impute', df.columns.values)
<<<<<<< HEAD
        non_numeric = [col for col in col_to_impute if not pd.api.types.is_numeric_dtype(df[col])]
=======
>>>>>>> a7aed5ab2ddd9770497f1227eb9b52b4f9bfece1
        impute_method = st.selectbox('Select a method', ['Simple Imputer', 'KNN', 'Iterative Impugter'])
        if impute_method == 'Simple Imputer':
            strategy = st.selectbox('Selecct strategy', ['mean', 'median', 'most_frequent', 'constant'])
            impute_btn = st.button('Fill Missing values')
            if impute_btn:
<<<<<<< HEAD
                if strategy != 'most_frequent' and non_numeric:
=======
                if strategy != 'most_frequent' and categorical_data:
>>>>>>> a7aed5ab2ddd9770497f1227eb9b52b4f9bfece1
                    st.error('Please select only numeric columns or use most frequent strategy')
                elif strategy and col_to_impute:
                    simple = SimpleImputer(strategy=strategy)
                    df[col_to_impute] = simple.fit_transform(df[col_to_impute])
                    st.session_state.df = df
                    st.write(df.isna().sum())
                    st.success('Missing values filled successfuly')
                else:
                    st.error('Please select a column and a strategy')
        elif impute_method == 'KNN':
            impute_btn = st.button('Fill missing values')
            if impute_btn:
<<<<<<< HEAD
                if col_to_impute in non_numeric:
=======
                if [col for col in col_to_impute if col in numeric_data] == col_to_impute: ## only numeric
>>>>>>> a7aed5ab2ddd9770497f1227eb9b52b4f9bfece1
                    knn = KNNImputer()
                    df[col_to_impute] = knn.fit_transform(df[col_to_impute])
                    st.session_state.df = df
                    st.write(df.isna().sum())
                    st.success('Missing values filled successfuly')
                else:
<<<<<<< HEAD
                    st.error('Please select a valid categorical column')
        elif impute_method == 'Iterative Impugter' and not non_numeric:
            impute_btn = st.button('Fill missing values')
            if impute_btn:
                if col_to_impute in non_numeric:
=======
                    st.error('Please select a valid numeric column')
        elif impute_method == 'Iterative Impugter' and numeric_data:
            impute_btn = st.button('Fill missing values')
            if impute_btn:
                if [col for col in col_to_impute if col in numeric_data] == col_to_impute: # only numeric
>>>>>>> a7aed5ab2ddd9770497f1227eb9b52b4f9bfece1
                    iterative = IterativeImputer()
                    df[col_to_impute] = iterative.fit_transform(df[col_to_impute])
                    st.session_state.df = df
                    st.write(df.isna().sum())
                    st.success('Missing values filled successfuly')
                else:
<<<<<<< HEAD
                    st.error('Please select a valid categorical column')
=======
                    st.error('Please select a valid numeric column')
>>>>>>> a7aed5ab2ddd9770497f1227eb9b52b4f9bfece1
    st.markdown('---')
    st.header('Encoding')
    col_to_encode = st.selectbox('Select column to encode it', categorical_data)
    encode_method = st.selectbox('Choose a method', ['Label', 'One hot']) 
    # other types of encoding will require more input from the user let it to futuer development
    encode_btn = st.button('Encode data')
    if encode_btn:
        if encode_method == 'Label' and col_to_encode != None:
            label = LabelEncoder()
            df[col_to_encode] = label.fit_transform(df[[col_to_encode]])
            st.session_state.df = df
            st.write(df[col_to_encode])
            st.success(f'{col_to_encode} encoded successfuly')
        elif encode_method == 'One hot' and col_to_encode != None:
            dummies = pd.get_dummies(df[col_to_encode])
            df.drop(col_to_encode, inplace=True, axis=1)
            df = pd.concat([df, dummies], axis=1)
            st.session_state.df = df
            st.write(df)
            st.success(f'{col_to_encode} encoded successfuly')
        else:
            st.error('Please select a column and a method')

    st.markdown('---')
    st.header('Scaling')
    col_to_scale = st.selectbox('Choose a column to scale', numeric_data)
    scale_method = st.selectbox('Method', ['MinMax scaler', 'Standared scaler'])
    scale_btn = st.button('Scale data')
    if scale_btn:
        if scale_method == 'MinMax scaler' and col_to_scale != None:
            scale_minmax = MinMaxScaler()
            df[col_to_scale] = scale_minmax.fit_transform(df[[col_to_scale]])
            st.session_state.df = df
            st.write(df[col_to_scale])
            st.success(f'{col_to_scale} scaled successfuly')
        elif scale_method == 'Standared scaler' and col_to_scale != None:
            scale_standard = StandardScaler()
            df[col_to_scale] = scale_standard.fit_transform(df[[col_to_scale]])
            st.session_state.df = df
            st.write(df[col_to_scale])
            st.success(f'{col_to_scale} scaled successfuly')
        else:
            st.error('Please select a column and a method')
    st.markdown('---')
    st.header('Outliers')
    st.subheader('Detect outliers')
    global outliers_indcies
    check_outlier_col = st.selectbox('Choose a column to detect ouliers', numeric_data)
    outlier_method = st.selectbox('Choose outlier detection method', ['Z-score', 'IQR'])
    threshold = st.number_input('Threshold', min_value=0.0, max_value=3.0, value=1.5)
    st.caption('Use the threshold to determine what above this number will be considered outlier ' \
    ' *note that this number is absloute value, so no negative numbers')
    outlier_btn = st.button('Check outliers')
    if outlier_btn:
        if check_outlier_col and outlier_method == 'Z-score':
            z_scores = abs(stats.zscore(df[check_outlier_col]))
            outliers_indcies = df[check_outlier_col][z_scores > threshold].index
            if len(outliers_indcies) > 0:
                st.write(df[check_outlier_col].loc[outliers_indcies])
                st.session_state['outliers_indcies'] = outliers_indcies
                st.write(f'Skew: {df[check_outlier_col].skew()}')
                st.caption('*note the nearest to zero is better')
            else:
                st.warning('No outliers in this column')
        elif check_outlier_col and outlier_method == 'IQR':
            Q1 = df[check_outlier_col].quantile(.25)
            Q3 = df[check_outlier_col].quantile(.75)
            IQR = Q3 - Q1
            outliers_indcies = df[check_outlier_col][(df[check_outlier_col] < Q1 - threshold * IQR) | (df[check_outlier_col] > Q3 + threshold * IQR)].index
            if len(outliers_indcies) > 0:
                st.write(df[check_outlier_col].loc[outliers_indcies])
                st.session_state['outliers_indcies'] = outliers_indcies
                st.write(f'Skew: {df[check_outlier_col].skew()}')
                st.caption('*note the nearest to zero is better')            
            else:
                st.warning('No outliers in this column')
                st.session_state['outliers_indcies'] = []
        else:
            st.error('Please select a column and a method')
    st.subheader('Deal with outliers')
    st.caption('This will reduce the effect of outliers')
    outlier_remove_method = st.selectbox('Choose a method to fix outliers', ['Drop', 'Log transform', 'Boxcox', 'Clip'])
    if outlier_remove_method == 'Log transform':
        trans_factor = st.number_input('Transform factor', min_value=1.0, value=1.0)
        st.session_state.trans_factor = trans_factor
    elif outlier_remove_method == 'Clip':
        st.caption('Limit the value to scpesefied range')
        col1, col2 = st.columns([1, 1])
        with col1:
            from_n = st.number_input('From', df[check_outlier_col].min(), df[check_outlier_col].max(), value=df[check_outlier_col].min(),
                            placeholder='From', key=0)
            st.session_state.from_n = from_n
        with col2:
<<<<<<< HEAD
            to_n = st.number_input('From', df[check_outlier_col].min(), df[check_outlier_col].max(), value=df[check_outlier_col].max(),
                            placeholder='From')
=======
            to_n = st.number_input('To', df[check_outlier_col].min(), df[check_outlier_col].max(), value=df[check_outlier_col].max(),
                            placeholder='To')
>>>>>>> a7aed5ab2ddd9770497f1227eb9b52b4f9bfece1
            st.session_state.to_n = to_n
    fix_outlier_btn = st.button('Fix outliers')
    if fix_outlier_btn:
        outliers_indcies = st.session_state.get('outliers_indcies', [])
        if outlier_remove_method == 'Drop':
            df.drop(index=outliers_indcies, axis=0, inplace=True)
            st.session_state.df = df
            st.session_state.outliers_indcies = []
            st.write(df)
            st.write(f'Skew: {df[check_outlier_col].skew()}')
            st.caption('*note the nearest to zero is better')
            st.success('Outliers droped successfuly')
        elif outlier_remove_method == 'Log transform':
            st.caption('This method helps to reduce the effect of outliers by modefying skewness using '
            ' transform factor and will not drictrly remove detected outliers')
            df[check_outlier_col] = df[check_outlier_col].apply(lambda x: np.log(x + st.session_state.trans_factor))
            st.session_state.df = df
            st.write(df)
            st.write(f'Skew: {df[check_outlier_col].skew()}')
            st.caption('*note the nearest to zero is better')
            st.success('Skew modefied successfuly')
        elif outlier_remove_method == 'Boxcox':
            st.caption('This method helps to reduce the effect of outliers by modefying skewness '
            ' and will not drictrly remove detected outliers')
            df[check_outlier_col], _ = stats.boxcox(df[check_outlier_col])
            st.session_state.df = df
            st.write(df)
            st.write(f'Skew: {df[check_outlier_col].skew()}')
            st.caption('*note the nearest to zero is better')
            st.success('Skew modefied successfuly')
        elif outlier_remove_method == 'Clip':
            st.caption('This method helps to reduce the effect of outliers by modefying skewness '
            ' and will not drictrly remove detected outliers')
            df[check_outlier_col] = df[check_outlier_col].clip(st.session_state.from_n, st.session_state.to_n)
            st.session_state.df = df
            st.write(df)
            st.write(f'Skew: {df[check_outlier_col].skew()}')
            st.caption('*note the nearest to zero is better')
            st.success('Skew modefied successfuly')
        else:
            st.error('Please select a column and a method.')

    st.markdown('---')
    st.header('Dimentionality reduction')
    reduce_method = st.selectbox('Method', ['PCA', 'RFE'])
    components = st.number_input('Components', min_value=1, max_value=len(df.columns), value=1)
    st.session_state.component = components
    reduce_btn = st.button('Reduce dimentionality')
    if reduce_btn:
        if reduce_method == 'PCA':
            pca = PCA(st.session_state.component)
            if df[numeric_data].isna().any().sum():
                st.error('You need to handle null values first')
            else:
                transformed = pca.fit_transform(df[numeric_data])
                transformed = pd.DataFrame(transformed)
                df.drop(numeric_data, axis=1, inplace=True)
                df = pd.concat([df, transformed], axis=1)
                st.session_state.df = df
                st.write(df)
                st.caption('This only')
                st.success('Dimentions reduced successfuly')
        elif reduce_method == 'RFE':
            # rfe = RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=2)
            # rfe = rfe.fit(X_std, y)
            # st.session_state.df = df
            # st.write(df[col_to_scale])
            # st.success('Dimentions reduced successfuly')
            st.warning('Future development')
        else:
            st.error('Please select a method')
<<<<<<< HEAD
    st.markdown('---')
    st.header('Split data')
=======
>>>>>>> a7aed5ab2ddd9770497f1227eb9b52b4f9bfece1




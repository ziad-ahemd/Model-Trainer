import streamlit as st
import numpy as np
import pandas as pd
from file_upload import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error


#---------------------------------- Classification ------------------------------------------------
def train_rf_model(X_train, y_train, X_test):
    from sklearn.ensemble import RandomForestClassifier

    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2', None]
    }


    # Grid search with 5-fold cross-validation
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )

    # Fit on training data
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Train best model on full training data
    best_model.fit(X_train, y_train)

    # Predict on test data
    y_pred = best_model.predict(X_test)

    return y_pred



def train_lr_model(X_train, y_train, X_test):
    from sklearn.linear_model import LogisticRegression

    # Define hyperparameter grid
    param_grid = {
        'penalty': ['l1', 'l2', 'elasticnet'],
        'solver': ['saga'],  # saga supports all penalty types
        'C': [0.01, 0.1, 1, 10],
        'l1_ratio': [0.0, 0.5, 1.0]  # used only when penalty = 'elasticnet'
    }

    # Grid search with 5-fold cross-validation
    grid_search = GridSearchCV(
        estimator=LogisticRegression(max_iter=1000, random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )

    # Fit on training data
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Train best model on full training data
    best_model.fit(X_train, y_train)

    # Predict on test data
    y_pred = best_model.predict(X_test)

    return y_pred



def train_SGD_model(X_train, y_train, X_test):
    from sklearn.linear_model import SGDClassifier

    # Define hyperparameter grid
    param_grid = {
        'penalty': ['l2', 'l1', 'elasticnet'],
        'loss': ['log_loss'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['optimal', 'invscaling', 'constant', 'adaptive'],
        'eta0': [0.01, 0.1],
        'l1_ratio': [0.0, 0.5, 1.0]
    }


    # Grid search with 5-fold cross-validation
    grid_search = GridSearchCV(
        estimator=SGDClassifier(max_iter=1000,random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )

    # Fit on training data
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Train best model on full training data
    best_model.fit(X_train, y_train)

    # Predict on test data
    y_pred = best_model.predict(X_test)

    return y_pred



def train_SVM_model(X_train, y_train, X_test):
    from sklearn.svm import SVC

    # Define hyperparameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto', 0.01, 0.001],
    }

    # Grid search with 5-fold cross-validation
    grid_search = GridSearchCV(
        estimator=SVC(),
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )

    # Fit on training data
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Train best model on full training data
    best_model.fit(X_train, y_train)

    # Predict on test data
    y_pred = best_model.predict(X_test)

    return y_pred


def train_DT_model(X_train, y_train, X_test):
    from sklearn.tree import DecisionTreeClassifier

    # Define hyperparameter grid
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 3, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None, 'sqrt', 'log2'],
    }

    # Grid search with 5-fold cross-validation
    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )

    # Fit on training data
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Train best model on full training data
    best_model.fit(X_train, y_train)

    # Predict on test data
    y_pred = best_model.predict(X_test)

    return y_pred


def train_KNN_model(X_train, y_train, X_test):
    from sklearn.neighbors import KNeighborsClassifier
    # Define valid hyperparameter grid for KNN
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    # Grid search with 5-fold cross-validation
    grid_search = GridSearchCV(
        estimator=KNeighborsClassifier(),
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )

    # Fit on training data
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Train best model on full training data
    best_model.fit(X_train, y_train)

    # Predict on test data
    y_pred = best_model.predict(X_test)

    return y_pred


def train_NN_model(X_train, y_train, X_test):
    from sklearn.neural_network import MLPClassifier
    # Define valid hyperparameter grid for KNN
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (100, 50), (50, 50)],
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [300]
    }

    # Grid search with 5-fold cross-validation
    grid_search = GridSearchCV(
        estimator=MLPClassifier(),
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )

    # Fit on training data
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Train best model on full training data
    best_model.fit(X_train, y_train)

    # Predict on test data
    y_pred = best_model.predict(X_test)

    return y_pred

#---------------------------------- Regression ------------------------------------------------
def call_ridge(x_train, x_test, y_train):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV
    param = {'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]}
    model = Ridge()
    grid = GridSearchCV(model, param_grid=param, cv=5, scoring='neg_mean_squared_error')
    grid.fit(x_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(x_test)

    return y_pred

def call_lasso(x_train, x_test, y_train):
    from sklearn.linear_model import Lasso
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import mean_squared_error

    param = {'alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0]}
    model = Lasso(max_iter=10000)
    grid = GridSearchCV(model, param_grid=param, cv=5, scoring='neg_mean_squared_error')
    grid.fit(x_train, y_train)

    best_model = grid.best_estimator_

    y_pred = best_model.predict(x_test)

    return y_pred

def call_sgd_lasso(x_train, x_test, y_train):
    from sklearn.linear_model import SGDRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import mean_squared_error


    param = {'alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0]}
    model = SGDRegressor(penalty='l1', max_iter=1000, tol=1e-3, random_state=42)
    grid = GridSearchCV(model, param_grid=param, cv=5, scoring='neg_mean_squared_error')
    grid.fit(x_train, y_train)

    best_model = grid.best_estimator_

    y_pred = best_model.predict(x_test)

    return y_pred

def call_xgboost_regressor(x_train, x_test, y_train):
    from xgboost import XGBRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import mean_squared_error

    param = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2,0.3]
    }

    model = XGBRegressor(random_state=42, objective='reg:squarederror')
    grid = GridSearchCV(model, param_grid=param, cv=5, scoring='neg_mean_squared_error', verbose=0)
    grid.fit(x_train, y_train)

    best_model = grid.best_estimator_
    print('Best parameters (XGBoost):', grid.best_params_)

    y_pred = best_model.predict(x_test)

    return y_pred

def show():
    st.title('Model traning')

    # Retreving dataframe
    if 'df' not in st.session_state:
        st.session_state.df = get_df()
    df = st.session_state.df
    if df.empty:
        st.warning("Dataset is empty, please upload a proper dataset")
        return
    st.dataframe(df)
    st.markdown('---')

    # Splitting data
    st.header('Splitting data')
    target_col = st.selectbox('Select a target', df.columns.values)
    y = df[target_col]
    X = df.drop(target_col, axis = 1)
    st.caption('Train sample precentage from 0 to 100')
    train_sample = st.number_input(label='Perecentage', min_value=1, max_value=99, value=20)
    split_btn = st.button('Split')
    if split_btn:
        if target_col:
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=train_sample/100, random_state=5)
            st.session_state.x_train = x_train
            st.session_state.x_test = x_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            col1, col2 = st.columns([1, 1])
            with col1:
                st.caption('Features training set')
                st.write(x_train)
                st.caption('Features testing set')
                st.write(x_test)
            with col2:
                st.caption('Target training set')
                st.write(y_train)
                st.caption('Target testing set')
                st.write(y_test)
        else:
            st.error('Please select a valid target')
    st.markdown('---')

    st.header('Model selection')
    select_model = st.selectbox('Choose a model', ['Random forest classifier', 'LogisticRegression', 'SGD classifier', 'Support vector machine',
                'Decision tree classifier', 'K neighbors classifier', 'MLPClassifier', 'Ridge', 'Lasso', 'SGD regressor',
                'XGB regressor'])
    st.header('Traning data')
    train_btn = st.button('Train')
    if train_btn:
        
        if select_model == 'Random forest classifier':
            y_predict = train_rf_model(st.session_state.x_train, st.session_state.y_train, st.session_state.x_test)
            st.session_state.y_predict = y_predict
            st.write(y_predict)
            st.success('Model trained successfuly')
        elif select_model == 'LogisticRegression':
            y_predict = train_lr_model(st.session_state.x_train, st.session_state.y_train, st.session_state.x_test)
            st.session_state.y_predict = y_predict
            st.write(y_predict)
            st.success('Model trained successfuly')
        elif select_model == 'SGD classifier':
            y_predict = train_SGD_model(st.session_state.x_train, st.session_state.y_train, st.session_state.x_test)
            st.session_state.y_predict = y_predict
            st.write(y_predict)
            st.success('Model trained successfuly')
        elif select_model == 'Support vector machine':
            y_predict = train_SVM_model(st.session_state.x_train, st.session_state.y_train, st.session_state.x_test)
            st.session_state.y_predict = y_predict
            st.write(y_predict)
            st.success('Model trained successfuly')
        elif select_model == 'Decision tree classifier':
            y_predict = train_DT_model(st.session_state.x_train, st.session_state.y_train, st.session_state.x_test)
            st.session_state.y_predict = y_predict
            st.write(y_predict)
            st.success('Model trained successfuly')
        elif select_model == 'K neighbors classifier':
            y_predict = train_KNN_model(st.session_state.x_train, st.session_state.y_train, st.session_state.x_test)
            st.session_state.y_predict = y_predict
            st.write(y_predict)
            st.success('Model trained successfuly')
        elif select_model == 'MLPClassifier':
            y_predict = train_NN_model(st.session_state.x_train, st.session_state.y_train, st.session_state.x_test)
            st.session_state.y_predict = y_predict
            st.write(y_predict)
            st.success('Model trained successfuly')
        elif select_model == 'Ridge':
            y_predict = call_ridge(st.session_state.x_train, st.session_state.x_test, st.session_state.y_train)
            st.session_state.y_predict = y_predict
            st.write(y_predict)
            st.success('Model trained successfuly')
        elif select_model == 'Lasso':
            y_predict = call_lasso(st.session_state.x_train, st.session_state.x_test, st.session_state.y_train)
            st.session_state.y_predict = y_predict
            st.write(y_predict)
            st.success('Model trained successfuly')
        elif select_model == 'SGD regressor':
            y_predict = call_sgd_lasso(st.session_state.x_train, st.session_state.x_test, st.session_state.y_train)
            st.session_state.y_predict = y_predict
            st.write(y_predict)
            st.success('Model trained successfuly')
        elif select_model == 'XGB regressor':
            y_predict = call_xgboost_regressor(st.session_state.x_train, st.session_state.x_test, st.session_state.y_train)
            st.session_state.y_predict = y_predict
            st.write(y_predict)
            st.success('Model trained successfuly')
        else:
            st.error('Please select a model')
    
def get_data():
    return st.session_state.get('y_predict', None), st.session_state.get('y_test', None)
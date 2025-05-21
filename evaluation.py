from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import streamlit as st
import model

def show():
    model_type=st.session_state.model_type
    y_preds=st.session_state.preds
    y_test=st.session_state.y_test
    classification = "classification"
    regression="regression"

    if model_type == classification:
        def classifier(model_type,y_preds, y_test):
        
            st.write(classification_report(y_test, y_preds))
    if model_type == regression:
        def regression(model_type,y_preds, y_test):
                mae = mean_absolute_error(y_test, y_preds)
                mse = mean_squared_error(y_test, y_preds)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_preds)

                st.write("MAE:", mae)
                st.write("MSE:", mse)
                st.write("RMSE:", rmse)
                st.write("R² Score:", r2)

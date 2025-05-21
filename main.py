import streamlit as st

st.set_page_config(
    page_title="Model trainer", page_icon='OIP.jpg', layout="wide", initial_sidebar_state="expanded"
)

import file_upload
import preprocessing
import visualization
import model
import evaluation



# Navbar
st.sidebar.title("Dashboard")

view = st.sidebar.radio(
    "Navigate to:", ["File upload", "Preprocessing", "Visualization", "Model", "Evalation"]
)




if view == "File upload":
    file_upload.show()
    
elif view == "Preprocessing":
    preprocessing.show()
    
elif view == "Visualization":
    visualization.show()
    
elif view == "Model":
    model.show()
    
elif view == "Evaluation":
    evaluation.show()
    



#I-M-P-O-R-T-S

import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import accuracy_score , confusion_matrix
import requests


#Logger
def log(message):
    timestamp = datetime.now().strftime("%Y-%M-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")



#Session-State Initialization
if "cleaned_saved" not in st.session_state:
    st.session_state.cleaned_saved =False


#Folder Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR , "data", "raw")
CLEANED_DIR = os.path.join(BASE_DIR , "data","cleaned")
os.makedirs(RAW_DIR , exist_ok=True)
os.makedirs(CLEANED_DIR,exist_ok=True)
log("Application Started")
log(f"Raw Directory - {RAW_DIR}")
log(f"Cleaned Directory - {CLEANED_DIR}")


#Page Config
st.set_page_config("End to End SVMs",layout="wide")
st.title("End to end SVM platform")

#Side bar : Model Settings

st.sidebar.header("SVM settings")
kernel = st.sidebar.selectbox("Kernel",["linear","rbf","poly","sigmoid"])
C = st.sidebar.slider("C [Regularization]", 0.01,10.0,1.0)
gamma = st.sidebar.selectbox("Gamma",["scale","auto"])

log(f"SVM settings ---> Kernel = {kernel} , c - {C} , Gamma : {gamma}")

#STEP 1: Data Ingestion
st.header("Step 1: Data Ingestion")
log("Step 1 started : Data Intgesion")

option = st.radio("Choose Data source " , ["Download Dataset" , "Upload CSV"])
df = None
raw_path = None

if option == "Download Dataset":
    if st.button("Download Iris Dataset"):
        log("Downloading iris dataset")
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
        response = requests.get(url)

        raw_path = os.path.join(RAW_DIR, "iris.csv")
        with open(raw_path,"wb") as f:
            f.write(response.content)
        
        df = pd.read_csv(raw_path)
        st.success("dataset downloaded sucessfully")
        log(f"Iris dataset saved at {raw_path}")

if option == "Upload CSV":
    uploaded_file = st.file_uploader("upload CSV file",type=["csv"])
    if uploaded_file:
        raw_path = os.path.join(RAW_DIR,uploaded_file.name)
        with open(raw_path , "wb") as f:
            f.write(uploaded_file.getbuffer())
        df=pd.read_csv(raw_path)
        st.success("file upload sucessfully")
        log(f"Uploaded data saved at {raw_path}")

#Step 2 : EDA

if df is not None:
    st.header("step 2: EDA")
    log("step 2 EDA staarted")

    st.dataframe(df.head())
    st.write("shape",df.shape)
    st.write("Missing Values :",df.isnull().sum())

    fig,ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only = True),annot=True , cmap = "Blues" , ax=ax)
    st.pyplot(fig)

    log("EDA Completed")

#Step 3 : Data Cleaning

if df is not None:
    st.header("Step 3: Data Cleaning")
    strategy = st.selectbox(
        "Missing value strategy",
        ["Mean" , "Median", "Drop Rows"]
    )
    df_clean = df.copy()

    if strategy == "Drop Rows":
        df_clean =df_clean.dropna()
    else:
        for col in df_clean.select_dtypes(include=np.number):
            if strategy == "Mean":
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    st.session_state.df_clean=df_clean
    st.success("data cleaning completed")

else:
    st.info("please complete step 1 (data ingestion) first...")


#Step 4 : save the cleaned data

if st.button("Save cleaned dataset"):
    if st.session_state.df_clean is None :
        st.error("no cleaned data form , complete step 2 first")
    else:
        timestamp = datetime.now().strftime("%Y%M%d_%H%M%S")
        cleaned_filename = (f"cleaned_dataset_{timestamp}.csv")
        clean_path = os.path.join(CLEANED_DIR , cleaned_filename)

        st.session_state.df_clean.to_csv(clean_path , index=False)
        st.success("Cleaned Dataset saved")
        st.info(f"Saved at : {clean_path}")
        log(f"Cleaned dataset saved at {clean_path}")


#Step 5 : load cleaned dataset

st.header("Step 5 : load cleaned dataset")
clean_files = os.listdir(CLEANED_DIR)

if not clean_files:
    st.warning("No cleaned datasets found , please save one in step 4..")
else:
    selected = st.selectbox("Select cleaned dataset",clean_files)
    df_model = pd.read_csv(os.path.join(CLEANED_DIR,selected))

    st.success(f"Loaded the dataset:{selected}")
    log(f"Loaded cleaned dataset : {selected}")

    st.dataframe(df_model.head())


#Step 6 : Train SVM

st.header("Step 6 : train SVM")
log("Step 6 started ")

tar = st.selectbox("Select target Columns",["species"])
y=df_model[tar]
if y.dtype == "object":
    y=LabelEncoder().fit_transform(y)
    log("Target column encoded")

#Select numeric features only

x=df_model.drop(columns=[tar])
x=x.select_dtypes(include = np.number)

if x.empty:
    st.error("No numeric features available for training")
    st.stop()


#Scale Features
scaler = StandardScaler()
x= scaler.fit_transform(x)

x_train , x_test , y_train , y_test = train_test_split(x,y, test_size = 0.25 , random_state = 42)


model = SVC(kernel=kernel)
model.fit(x_train,y_train)

#Evaluation metrics
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)

st.success(f"Accuracy : {acc:.2f}")
log(f"SVM trained succesfully | Accuracy = {acc : .2f}")

cm = confusion_matrix(y_test,y_pred)
fig,ax = plt.subplots()
sns.heatmap(cm , annot =True , fmt = "d" , cmap="Blues" ,ax=ax)
st.pyplot(fig)


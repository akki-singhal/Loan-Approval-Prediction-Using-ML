import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

#Funcation for cleaning spaces
def clean_data(st):
    st = st.strip()
    return st

#Data Reading
data = pd.read_csv('loan_approval_dataset.csv')

#Data Refining
data.drop(columns = ['loan_id'], inplace=True)
data.columns = data.columns.str.strip()
data['Assets'] = data.residential_assets_value + data.commercial_assets_value + data.luxury_assets_value + data.bank_asset_value
data.drop(columns = ['residential_assets_value','commercial_assets_value','luxury_assets_value','bank_asset_value'], inplace = True)
data.education = data.education.apply(clean_data)
data['education'] = data['education'].replace(['Graduate', 'Not Graduate'],[1,0])
data.self_employed = data.self_employed.apply(clean_data)
data.self_employed = data.self_employed.replace(['No', 'Yes'],[0,1])
data.loan_status = data.loan_status.apply(clean_data)
data.loan_status = data.loan_status.replace(['Approved', 'Rejected'],[1,0])

#Train Test Split
input_data = data.drop(columns=['loan_status'])
output_data = data['loan_status']
x_train,x_test,y_train,y_test = train_test_split(input_data,output_data, test_size=0.2, random_state=1)

#Data Preprocessing
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

#Model
model = LogisticRegression()
model.fit(x_train_scaled, y_train)

#Web App
st.header('Loan Predcition App')

no_of_dep = st.slider('Choose No of dependents', 0, 5)
grad = st.selectbox('Choose Education',['Graduate','Not Graduate'])
self_emp = st.selectbox('Self Emoployed ?',['Yes','No'])
Annual_Income = st.slider('Choose Annual Income', 0, 10000000)
Loan_Amount = st.slider('Choose Loan Amount', 0, 10000000)
Loan_Dur = st.slider('Choose Loan Duration', 0, 20)
Cibil = st.slider('Choose Cibil Score', 0, 1000)
Assets = st.slider('Choose Assets', 0, 10000000)

if grad =='Graduate':
    grad_s = 1
else:
    grad_s = 0

if self_emp =='No':
    emp_s = 0
else:
    emp_s = 1

if st.button("Predict"):
    pred_data = pd.DataFrame([[no_of_dep,grad_s,emp_s,Annual_Income,Loan_Amount,Loan_Dur,Cibil,Assets]],
                         columns=['no_of_dependents','education','self_employed','income_annum','loan_amount','loan_term','cibil_score','Assets'])
    
    #Standardization of input data
    pred_data = scaler.transform(pred_data)
    predict = model.predict(pred_data)
    if predict[0] == 1:
        st.markdown('Loan Is Approved')
    else:
        st.markdown('Loan Is Rejected')
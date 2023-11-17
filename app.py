import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('Web Development of medical diagnostic app')
st.subheader('Is the person diabetic ?')

df = pd.read_csv('diabetes.csv')
if st.sidebar.checkbox('View Data', False):
    st.write(df)

if st.sidebar.checkbox('View Distributions', False):
    df.hist()
    plt.tight_layout()
    st.pyplot()
    
    
    # create user interface
# step1 - load the pickled model
model = open('rfc.pickle', 'rb')
clf = pickle.load(model)
model.close()

# step2 = get the frontend user input
pregs = st.number_input('Pregnancies',0,17,0)
glucose = st.number_input( 'Glucose',44,199,44)
bp = st.slider('BloodPressure',20,140,20)
skin = st.slider('SkinThickness',7,99,7)
insulin = st.slider('Insulin',14,840,14)
bmi = st.slider('BMI',18,67,18)
dpf = st.slider('DiabetesPedigreeFunction',0.05,2.50,0.05)
age = st.slider('Age',21,85,21)

# step 3: convert user iput to model input

data = {'Pregnancies': pregs,
        'Glucose': glucose, 
        'BloodPressure': bp,
        'SkinThickness': skin,
        'Insulin': insulin ,
       'BMI':bmi, 'DiabetesPedigreeFunction':dpf, 'Age' :age,
}
input_data = pd.DataFrame([data])

# step 4: Get the prediction and get the result

prediction = clf.predict(input_data)[0]

if st.button('Predict'):
    if prediction ==1 :
        st.subheader('Diabetic')
    else : 
        st.subheader('Healthy')

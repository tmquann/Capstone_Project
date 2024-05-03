import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.tree import DecisionTreeRegressor as dtr

from sklearn.model_selection import train_test_split
import joblib
import matplotlib.dates as mdates

#setting the deplot tittle

st.title("Predict Quan Tesla Adj Close Stock Price")

# disable warning:
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the data
df = pd.read_csv('TESLA.csv')

df['Date'] = pd.to_datetime(df['Date']) 

#Assuming X and y represent your features and target variable
X = df.drop(columns=['Date', 'Close', 'Volume','Adj Close'])  
y = df['Adj Close'] #is our target variable need to show


#Define  Regressor model
RegModel1 = dtr()

#Train the model on 100% of the available data
RegModel1.fit(X, y)


#make streamlit app

def main():
    st.title("Predict Tesla Adj Close Stock Price")

    # Input form
    st.sidebar.subheader("Scroll for predict price here:")
    # Example input fields
    open_value = st.sidebar.slider('Open', min_value=float(df['Open'].min()), max_value=float(df['Open'].max()), value=float(df['Open'].mean()))
    high_value = st.sidebar.slider('High', min_value=float(df['High'].min()), max_value=float(df['High'].max()), value=float(df['High'].mean())) 
    low_value = st.sidebar.slider('Low', min_value=float(df['Low'].min()), max_value=float(df['Low'].max()), value=float(df['Low'].mean())) 

    # Predict button
    if st.sidebar.button('Predict'):
        # Make prediction
        prediction = predict_stock_price(open_value, high_value, low_value)

#Display predict
        st.subheader("Here what we predicted for our future Adj Close stocks:")
        st.write(prediction)


#Function to predict
def predict_stock_price(open_value, high_value, low_value):
    # Make predictions
    prediction = RegModel1.predict(np.array([[open_value, high_value, low_value]]))
    return prediction
main()
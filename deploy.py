import streamlit as st
import numpy as np
import pandas as pd
import random

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.title('Cricket Score Predictor!!')
st.caption('Get your score prediction by answering few questions...')


url = './t20.csv'
dataset = pd.read_csv(url)
X = dataset[['runs', 'wickets', 'overs', 'striker', 'non-striker']].values
y = dataset.iloc[:, 14].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

runs = st.number_input('Enter number of runs scored (Integer Value Only)')
wickets = st.number_input(
    'Enter the number of wickets down (Integer Value Only)')
overs = st.number_input(
    'Enter the number of completed overs (Can be a decimal value)')
striker = st.text_input('Enter the striker name')
non_striker = st.text_input('Enter the non-striker name')

start_prediction = False
if st.button('Predcit Score'):
    new_values = np.array(
        [[int(runs), int(wickets), overs, random.randint(0, 10), random.randint(0, 10)]])
    params = sc.transform(new_values)
    start_prediction = True
else:
    st.write(
        'Click on the above button to predict the score and wait until the result shows up!!')

model_ready = False
reg = RandomForestRegressor(n_estimators=100, max_features=None)
reg.fit(X_train, y_train)
model_ready = True

disabled = True

if runs and wickets and overs and striker and non_striker:
    disabled = False


if start_prediction and model_ready:
    pred = reg.predict(params)
    st.metric(label="Predicted Score : ", value=int(pred[0]))
    start_prediction = False

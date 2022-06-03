import streamlit as st
import pandas as pd
# missing data library
# import missingno as msno
import numpy as np
# import csv
# import matplotlib.pyplot as plt
from pandas import read_csv
import os.path
import pandas as pd
import os
from plotly import graph_objs as go
from fbprophet import Prophet
import pickle
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt

uploaded_file = st.file_uploader("Choose a dataset")


def predict_fb(df,time_column):
    columns = df.columns[df.columns != time_column]
    # columns.remove('Time')

    predict_column = st.selectbox(
        'Select column to predict',
        (columns))

    st.write('You selected:', predict_column)

    # options = ['pm25', 'pm1', 'pm10']

    options = [column for column in df.columns.tolist()]
    options.remove(time_column)

    options.remove(predict_column)
    print('selected to predict ', columns)
    print('the rest of columns', options)

    #
    # df['Time'] = pd.to_datetime(df['Time'], format="%Y-%m-%d %H:%M:%S")

    df.index = df[time_column]
    df.index.sort_values()
    df = df.interpolate(method='linear', axis=0).ffill().bfill()
    #     todo check predict
    #     df['y']=df['Time']
    #     df['ds']=df[predict_column]
    df.rename(columns={time_column: 'ds', predict_column: 'y'}, inplace=True)

    # df[['Time',predict_column]].rename({'Time':'ds',predict_column:'y'}, inplace=True)
    print('df is ', df.columns)
    print('df is ', df.head().to_string())
    # df_final = df[['TimeStamp', 'pm25', 'pm1', 'pm10']].rename({'TimeStamp': 'ds', predict_column: 'y'},
    #                                                                       axis='columns')

    # df.set_index('ds')[['y', options[0], options[1]]].plot()  # pm25,pm1,pm10
    chosen_percent = 80
    eighty_percent = int(chosen_percent / 100 * len(df))
    #
    train_df = df[:eighty_percent]
    print('train_df ', len(train_df))
    test_df = df[eighty_percent:]
    print('test_df ', len(test_df))

    model = Prophet(interval_width=0.9)
    for option in options:
        model.add_regressor(option, standardize=False)
    # model = Prophet(interval_width=0.9)
    # model.add_regressor(options[0], standardize=False)
    # model.add_regressor(options[1], standardize=False)
    model.fit(train_df)
    #
    forecast_initial = model.predict(test_df)
    forecast = forecast_initial[['ds', 'yhat']]

    forecast = forecast.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    result = pd.concat((forecast['yhat'], test_df), axis=1)
    result = result.rename({'yhat': 'predicted ' + str(predict_column), 'y': predict_column, 'ds': 'date'},
                           axis='columns')
    result['date'] = pd.to_datetime(result['date'])
    result['predicted-actual'] = result['predicted ' + str(predict_column)] - result[predict_column]
    st.write("dataframe is \n ")
    st.dataframe(result)


if uploaded_file is not None:  # read csv …
    print("uploaded")

    df = pd.read_csv(uploaded_file)
    print('df is ', df.head())
    # df = df.interpolate(method='linear', axis=0).ffill().bfill()

    time_column = st.selectbox(
        'Select Time column',
        (df.columns))

    st.write('You selected:', time_column)
    st.write(df)
    predict_fb(df,time_column)

else:
    print("no csv")

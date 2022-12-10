# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
!pip install yfinance
import yfinance as yf
from datetime import date
from pandas_datareader import data as pdr
yf.pdr_override()

# deep learning libraries
import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os


# fetch data for training
def SaveData(df, filename):
  df.to_csv(filename + '.csv')
today = date.today()
dataname='^NSEI'+'_'+str(today)
data=pdr.get_data_yahoo('^NSEI', start='2015-01-01', end=today)
SaveData(data, dataname)
data= pd.read_csv(dataname+'.csv')

# drop not required column
data.drop(columns=['Adj Close'], inplace=True)
data['Date'] = pd.to_datetime(data['Date'])

# 80% will be used for traning, and 20% for testing
train_size = 0.8       # 80%
split_index = int(train_size * data.shape[0])

factors_column = ['Open', 'High', 'Low', 'Close', 'Volume']
y_col_index = 3 # Close

train_set = data[factors_column].values[:split_index]
test_set = data[factors_column].values[split_index:]

# scale our price from 0 to 1
sc = MinMaxScaler(feature_range = (0, 1))
train_set_scaled = sc.fit_transform(train_set)
test_set_scaled = sc.fit_transform(test_set)

# Predicting Closing Price
# Generate windowed timestamp data
# this function will combine data of 20 days (we can change it using time_window parameter)
time_window = 28      # 60 days
days_step =1         # skip 1 days in between, can be set to 1 day

# later...

# load json and create model
json_file = open('model_5_28.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_5_28.h5")
print("Loaded model from disk")


## future forecasting
def predict_n_days(df, n=30, step=0.1):
    for i in range(n):
        X = df[factors_column].values[df.shape[0] - time_window::step]
        X = sc.transform(X)
        y = loaded_model.predict(np.expand_dims(X, axis=0))
        y = sc.inverse_transform(y)[0]

        next_day_prediction = {key: value for key, value in zip(factors_column, y)}
        next_day_prediction['Date'] = df.iloc[-1].Date + pd.Timedelta(days=1)

        df = df.append(next_day_prediction, ignore_index=True)
    return df


days = 5
# days_step =0.1
predicted_df = predict_n_days(data, days, days_step)

# evaluate loaded model on test data
#loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#score = loaded_model.evaluate(X_train,y_train, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))

fig = plt.figure(figsize=(25, 7))
# plt.plot(data.Date, data.Close, 'r-', label = 'Option Close')
plt.plot(predicted_df.Date.values[-days:], predicted_df.Close[-days:], 'b--', label='Future Prediction')

plt.title('NIFTY50 Prediction')
plt.xlabel('Date')
plt.ylabel('NIFTY50')
plt.legend()
plt.show()
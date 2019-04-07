import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime

dataset_path = "dataset.csv"
# Import data
#data = pd.read_csv('data_stocks.csv')
column_names = ["ASPFWR5","US10YR","EPS","PER","OPEN","HIGH","LOW","CLOSE","BDIY","VIX","PCR","MVOLE","DXY","ASP","ADVDECL","FEDFUNDS","NYSEADV","IC","BAA","NOS","BER","DVY","PTB","AAA","SI","URR","FOMC","PPIR","RV","LOAN","VVIX","NAPMNEWO","NAPMPRIC","NAPMPMI","US3M","DEL","BBY","HTIME","LTIME","TYVIX","PUC","CRP","TERM","UR","INDPRO","HS","VRP","CAPE","CATY","INF","SIM","TOM","RELINF","DTOM","sentiment1","sentiment2","sentiment3","Hulbert.sentiment"]
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "NA", comment='\t',index_col = 0,
                      parse_dates = True ,sep=",", skipinitialspace=True)

raw_dataset = raw_dataset.dropna()
dataset = raw_dataset['20100101':'20170101']
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout


#Data cleaning
training_set=dataset['ASPFWR5']
training_set=pd.DataFrame(training_set)
# Feature Scaling Normalization
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])

X_train , y_train = np.array(X_train), np.array(y_train)
# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# Initialising the RNN
regressor = Sequential()
# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))
# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 5, batch_size = 32)


dataset_test = raw_dataset['20170101':]
test_set=dataset_test['ASPFWR5']
test_set=pd.DataFrame(test_set)
real_stock_price = dataset_test.iloc[:, 1:2].values
# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset['ASPFWR5'], dataset_test['ASPFWR5']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
predicted_stock_price=pd.DataFrame(predicted_stock_price)

plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
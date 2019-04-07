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
from tensorflow.keras.layers import CuDNNLSTM
from tensorflow.keras.layers import Dropout

train_days = 60
#Data cleaning
training_set=dataset[['ASPFWR5',"BDIY","VIX"]]
training_set=pd.DataFrame(training_set)
# Feature Scaling Normalization
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
# Creating a data structure with train_days timesteps and 1 output
X_train = []
y_train = []
for i in range(train_days,training_set_scaled.shape[0]):
    X_train.append(training_set_scaled[i-train_days:i, :])
    y_train.append(training_set_scaled[i, 0])

X_train , y_train = np.array(X_train), np.array(y_train)
# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))


# Initialising the RNN
regressor = Sequential()
# Adding the first CuDNNLSTM layer and some Dropout regularisation
regressor.add(CuDNNLSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 3)))
regressor.add(Dropout(0.2))
# Adding a second CuDNNLSTM layer and some Dropout regularisation
regressor.add(CuDNNLSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third CuDNNLSTM layer and some Dropout regularisation
regressor.add(CuDNNLSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth CuDNNLSTM layer and some Dropout regularisation
regressor.add(CuDNNLSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))
# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


dataset_test = raw_dataset['20170101':]
real_stock_price = dataset_test.iloc[:, 0].values
test_set=dataset_test['ASPFWR5']
test_set=pd.DataFrame(test_set)
dummy = dataset_test.copy()
dummy = dummy[['ASPFWR5',"BDIY","VIX"]]
dummy = sc.transform(dummy)
dummy = pd.DataFrame(dummy)

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset[['ASPFWR5',"BDIY","VIX"]], dataset_test[['ASPFWR5',"BDIY","VIX"]]), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - train_days:]
inputs = sc.transform(inputs)
X_test = []
for i in range(train_days, train_days + test_set.shape[0]):
    X_test.append(inputs[i-train_days:i, :])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))
predicted_stock_price = regressor.predict(X_test)
dummy[0] = predicted_stock_price
dummy = sc.inverse_transform(dummy)
dummy = pd.DataFrame(dummy)
predicted_stock_price = dummy[0] 
predicted_stock_price=pd.DataFrame(predicted_stock_price)

plt.plot(real_stock_price, color = 'red', label = 'Real ASPFWR5')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted ASPFWR5')
plt.title('ASPFWR5 Prediction')
plt.xlabel('Time')
plt.ylabel('ASPFWR5')
plt.legend()
plt.show()
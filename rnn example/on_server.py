import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime

from tensorflow.keras.models import load_model
regressor = load_model('model')

dataset_path = "dataset.csv"
# Import data
#data = pd.read_csv('data_stocks.csv')
raw_dataset = pd.read_csv(dataset_path, 
                      na_values = "NA", comment='\t',index_col = 0,
                      parse_dates = True ,sep=",", skipinitialspace=True)

dataset = raw_dataset.dropna()
dataset = raw_dataset['20100101':'20171220']
trans_set=dataset[['ASPFWR5']]
trans_set=pd.DataFrame(trans_set)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
trans_set_scaled = sc.fit_transform(trans_set)

def parse_dates():
    date_file = open("date.txt","r")
    date = date_file.read()
    date_file.close()
    date = pd.to_datetime(date)
    return date

date = parse_dates()

def recursion():
    if base_case:
        raw_dataset.tail(60)


dataset_test = raw_dataset['20170101':]
test_set=dataset_test['ASPFWR5']
test_set=pd.DataFrame(test_set)
dummy = dataset_test.copy()
dummy = dummy[['ASPFWR5',"BDIY","VIX"]]
dummy = sc.transform(dummy)
dummy = pd.DataFrame(dummy)

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset[['ASPFWR5',"BDIY","VIX"]], dataset_test[['ASPFWR5',"BDIY","VIX"]]), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:]
inputs = sc.transform(inputs)
X_test = []

X_test.append(inputs[i-60:i, :])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))
predicted_stock_price = regressor.predict(X_test)
dummy[0] = predicted_stock_price
dummy = sc.inverse_transform(dummy)
dummy = pd.DataFrame(dummy)
predicted_stock_price = dummy[0] 
predicted_stock_price=pd.DataFrame(predicted_stock_price)
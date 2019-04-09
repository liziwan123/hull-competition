import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model
regressor = load_model('model')

dataset_path = "dataset.csv"
# Import data
raw_dataset = pd.read_csv(dataset_path, 
                      na_values = "NA", comment='\t',index_col = 0,
                      parse_dates = True ,sep=",", skipinitialspace=True)

raw_dataset= raw_dataset[['ASPFWR5',"BDIY","VIX"]]
raw_dataset = raw_dataset['20100101':]


dataset = raw_dataset.dropna()
dataset = dataset['20100101':'20171220']
trans_set=dataset[['ASPFWR5',"BDIY","VIX"]]
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


def get_index(d):
        index = raw_dataset.index
        for i in range(0,len(raw_dataset.index)):
                if index[i] == d:
                        return i


def predict(index):
        data = raw_dataset[index-60:index]
        data = np.array(data)
        data = sc.transform(data)
        data = np.reshape(data,(1,data.shape[0],data.shape[1]))
        predicted_stock_price = regressor.predict(data)[0][0]
        p = pd.DataFrame([predicted_stock_price,0,0]).transpose()
        p = sc.inverse_transform(p) 
        return p[0][0]

for i in range(0,5):
        i_d = get_index(date) - (5 - i)
        raw_dataset.iloc[i_d,0] = predict(i_d)

print(predict(get_index(date)))


from __future__ import absolute_import, division, print_function

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

dataset_path = "dataset.csv"
column_names = ["ASPFWR5","US10YR","EPS","PER","OPEN","HIGH","LOW","CLOSE","BDIY","VIX","PCR","MVOLE","DXY","ASP","ADVDECL","FEDFUNDS","NYSEADV","IC","BAA","NOS","BER","DVY","PTB","AAA","SI","URR","FOMC","PPIR","RV","LOAN","VVIX","NAPMNEWO","NAPMPRIC","NAPMPMI","US3M","DEL","BBY","HTIME","LTIME","TYVIX","PUC","CRP","TERM","UR","INDPRO","HS","VRP","CAPE","CATY","INF","SIM","TOM","RELINF","DTOM","sentiment1","sentiment2","sentiment3","Hulbert.sentiment"]
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "NA", comment='\t',index_col = 0,
                      parse_dates = True ,sep=",", skipinitialspace=True)





dataset = raw_dataset.copy()

dataset = dataset[["ASPFWR5", "BDIY", "VIX", "SIM","TOM",]]
dataset = dataset.dropna()
train_dataset = dataset['20100101':'20170101']
test_dataset = dataset['20170101':]



train_stats = train_dataset.describe()
train_stats.pop("ASPFWR5")
train_stats = train_stats.transpose()


train_labels = train_dataset.pop('ASPFWR5')
test_labels = test_dataset.pop('ASPFWR5')

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

def build_model():
  model = keras.Sequential([
    layers.Dense(128, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model

model = build_model()

print(model.summary())



class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 2000

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])


loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])

plt.show()
"""
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()


plot_history(history)

model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                   validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

#plot_history(history)
"""



#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      14014
#
# Created:     05.04.2019
# Copyright:   (c) 14014 2019
# Licence:     <your licence>
#-------------------------------------------------------------------------------

def main():
    pass

if __name__ == '__main__':
    main()

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
dataset_path = "dataset.csv"
# Import data
#data = pd.read_csv('data_stocks.csv')
column_names = ["ASPFWR5","US10YR","EPS","PER","OPEN","HIGH","LOW","CLOSE","BDIY","VIX","PCR","MVOLE","DXY","ASP","ADVDECL","FEDFUNDS","NYSEADV","IC","BAA","NOS","BER","DVY","PTB","AAA","SI","URR","FOMC","PPIR","RV","LOAN","VVIX","NAPMNEWO","NAPMPRIC","NAPMPMI","US3M","DEL","BBY","HTIME","LTIME","TYVIX","PUC","CRP","TERM","UR","INDPRO","HS","VRP","CAPE","CATY","INF","SIM","TOM","RELINF","DTOM","sentiment1","sentiment2","sentiment3","Hulbert.sentiment"]
dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "NA", comment='\t',index_col = 0,
                      parse_dates = True ,sep=",", skipinitialspace=True)
# Drop date variable
#data = data.drop(['DATE'], 1)
#[["ASPFWR5","CLOSE" ,"EPS","PER","TOM","SIM","BDIY", "VIX","FEDFUNDS"]]
data = dataset.dropna()
train_dataset = dataset['20120101':'20170101']
test_dataset = dataset['20170101':]

# Dimensions of dataset
n = data.shape[0]
p = data.shape[1]
# Make data a numpy array
data = data.values



# Training and test data
train_start = 0
train_end = int(np.floor(0.95*n))
test_start = train_end
test_end = n
data_train = data[np.arange(train_start, train_end), :]
data_test = data[np.arange(test_start, test_end), :]



# Scale data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_train = scaler.fit_transform(data_train)
data_test = scaler.transform(data_test)
# Build X and y
X_train = data_train[:, 1:]
y_train = data_train[:, 0]
X_test = data_test[:, 1:]
y_test = data_test[:, 0]



# Import TensorFlow
import tensorflow as tf

# Define a and b as placeholders
a = tf.placeholder(dtype=tf.int8)
b = tf.placeholder(dtype=tf.int8)

# Define the addition
c = tf.add(a, b)





# Initializers
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()



# Model architecture parameters
n_stocks = p - 1
n_neurons_1 = 2048
n_neurons_2 = 1024
n_neurons_3 = 512
n_neurons_4 = 128
n_target = 1
# Layer 1: Variables for hidden weights and biases
W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
# Layer 2: Variables for hidden weights and biases
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
# Layer 3: Variables for hidden weights and biases
W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
# Layer 4: Variables for hidden weights and biases
W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))

# Output layer: Variables for output weights and biases
W_out = tf.Variable(weight_initializer([n_neurons_4, n_target]))
bias_out = tf.Variable(bias_initializer([n_target]))



 # Placeholder
X = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks])
Y = tf.placeholder(dtype=tf.float32, shape=[None])



# Hidden layer
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))

# Output layer (must be transposed)
out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))




# Cost function
mse = tf.reduce_mean(tf.squared_difference(out, Y))




# Optimizer
opt = tf.train.AdamOptimizer().minimize(mse)




# Make Session
net = tf.Session()
# Run initializer
net.run(tf.global_variables_initializer())

# Setup interactive plot
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(y_test)
line2, = ax1.plot(y_test*0.5)
plt.show()

# Number of epochs and batch size
epochs = 100
batch_size = 512

for e in range(epochs):

    # Shuffle training data
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    X_train = X_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    # Minibatch training
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = X_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]
        # Run optimizer with batch
        net.run(opt, feed_dict={X: batch_x, Y: batch_y})

        # Show progress
        if np.mod(i, 5) == 0:
            # Prediction
            pred = net.run(out, feed_dict={X: X_test})
            line2.set_ydata(pred)
            plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
            file_name = 'img/epoch_' + str(e) + '_batch_' + str(i) + '.png'
            plt.savefig('file_name')
<<<<<<< HEAD
            plt.pause(0.001)
=======
            plt.pause(0.01)
>>>>>>> a532657e728ad1861acb61fdebe9aa7271e349a3


# Print final MSE after Training
mse_final = net.run(mse, feed_dict={X: X_test, Y: y_test})
print(mse_final)







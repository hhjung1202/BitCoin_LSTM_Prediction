import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
tf.set_random_seed(777)  # reproducibility


def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

f = open('data_revise.csv', 'r')
rdr = csv.reader(f )
prices = []
for line in rdr:
    line = [float(i) for i in line]
    prices.append(line)
    print(line)
f.close()

# train Parameters
seq_length = 5
data_dim = 9
hidden_dim = 10
output_dim = 1
learning_rate = 0.001
iterations = 10000

# Open, High, Low, Volume, Close
xy = prices
xy = xy[::-1]
xy = MinMaxScaler(xy) #normalize
x = xy
y = xy[:, [-1]] # Close as label

# build a dataset
dataX = []
dataY = []
print(len(y), len(x))
for i in range(0, len(y) - seq_length):
    _x = x[i:i + seq_length]
    _y = y[i + 5]  # Next close price
    print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)

# train/test split
train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(
    dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(
    dataY[train_size:len(dataY)])

# input place holders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, output_dim])

# build a LSTM network
cell = tf.contrib.rnn.BasicLSTMCell(
    num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)

outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

#predict repression
Y_pred = tf.contrib.layers.fully_connected(
    outputs[:, output_dim], output_dim, activation_fn=None)  # We use the last cell's output

# cost/loss
loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# RMSE
targets = tf.placeholder(tf.float32, [None, output_dim])
predictions = tf.placeholder(tf.float32, [None, output_dim])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], feed_dict={
                                X: trainX, Y: trainY})
        print("[step: {}] loss: {}". format(i, step_loss))

    # Test step
    test_predict = sess.run(Y_pred, feed_dict={X: testX})
    rmse_val = sess.run(rmse, feed_dict={
                    targets: testY, predictions: test_predict})
    print("RMSE: {}".format(rmse_val))

    # Plot predictions
    print(testY)
    print(test_predict)
    plt.plot(testY , label= 'actual' )
    plt.plot(test_predict, label = 'predict')
    plt.xlabel("Time Period")
    plt.ylabel("coin Price")
    plt.show()


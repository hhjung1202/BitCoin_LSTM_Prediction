import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
import requests
import wget
import os.path
import csv

tf.set_random_seed(777)  # reproducibility

class Learning:
	def __init__(self):
		# AccPrice AccVolume open low high close ma boll rsi
		self.data = []
		self.seq_length = 5
		self.data_dim = 5
		self.hidden_dim = 10
		self.output_dim = 1
		self.learning_rate = 0.001
		self.iterations = 500



		self.x = []
		self.y = []

	def DataLoading(self):
		with open('data2.csv', newline="") as csvfile:
			spam = csv.reader(csvfile, delimiter=" ", quotechar="|")
			for row in spam:
				f = row[0].split(',')
				self.data.append(f[1:])

		
		xy = self.data
		xy = xy[::-1]
		xy = self.MinMaxScaler(xy) #normalize
		self.x = xy
		self.y = xy[:, [5]] # Close as label

	def MinMaxScaler(self, data):
		numerator = data - np.min(data, 0)
		denominator = np.max(data, 0) - np.min(data, 0)
		# noise term prevents the zero division
		return numerator / (denominator + 1e-7)

	def build_arch(self):
		with tf.variable_scope('LSTM1_layer'):
            # Conv1, [batch_size, 20, 20, 256]
			cell = tf.contrib.rnn.BasicLSTMCell(
				num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
			self.lstm_output, self.lstm_states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
			
            assert self.lstm_output.get_shape() == [cfg.batch_size, 20, 20, 256]

        # Primary Capsules layer, return [batch_size, 1152, 8, 1]
        with tf.variable_scope('Fully_conn_layer1'):
            Y_pred = tf.contrib.layers.fully_connected(self.lstm_output[:, output_dim], output_dim, activation_fn=None)
            assert Y_pred.get_shape() == [2]

    def loss(self):
    	

	def run(self):



# build a dataset
dataX = []
dataY = []
print(len(y), len(x))
for i in range(0, len(y) - seq_length):
	_x = x[i:i + seq_length]
	_y = y[i + seq_length]  # Next close price
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
#Y_pred = tf.contrib.layers.fully_connected(
#   outputs[:, output_dim], output_dim, activation_fn=None)  # We use the last cell's output
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
	plt.ylabel("Stock Price")
	plt.show()


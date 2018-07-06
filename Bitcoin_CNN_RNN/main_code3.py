import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

tf.set_random_seed(777)  # reproducibility
# conv - relu - lstm
class Learning:
    def __init__(self):
        # AccPrice AccVolume open low high close ma boll rsi
        self.seq_length = 9
        self.data_dim = 9
        self.hidden_dim = 16
        self.fn_output1 = 8
        self.fn_output2 = 5
        self.output_dim = 1
        self.learning_rate = 0.001
        self.epoch_num = 10000
        self.num_stacked_layers = 2
        self.drop_prob = 0.2

        self.trainX = []
        self.trainY = []
        self.textX = []
        self.textY = []

        self.train_error_summary = []
        self.test_error_summary = []
        self.test_predict = ''  # prediction with test data

        self.DataLoading()
        tf.reset_default_graph()
        self.X = tf.placeholder(tf.float32, [None, self.seq_length, self.data_dim, 1])
        self.Y = tf.placeholder(tf.float32, [None, self.output_dim])

    def DataLoading(self):
        data = []

        with open('./data_revise.csv', newline="") as csvfile:
        # with open('data_temp.csv', newline="") as csvfile:
            spam = csv.reader(csvfile)
            for row in spam:
                row = [float(i) for i in row]
                data.append(row)
        data = data[::-1]
        data = self.MinMaxScaler(data)  # normalize
        x = data
        y = data[:, [-1]]  # Close as label
        dataX = []
        dataY = []

        for i in range(0, len(y) - self.seq_length):
            _x = x[i:i + self.seq_length]
            _y = y[i + 5]  # Next close price
            #print(_x, "->", _y)
            dataX.append(_x)
            dataY.append(_y)

        dataX_np = np.array(dataX)
        dataX_reshaped = np.reshape(dataX_np, (dataX_np.shape[0], dataX_np.shape[1], dataX_np.shape[2], 1))
        print(dataX_reshaped.shape)
        train_size = int(len(dataY) * 0.7)
        test_size = len(dataY) - train_size
        self.trainX, self.testX = np.array(dataX_reshaped[0:train_size]), np.array(dataX_reshaped[train_size:len(dataX)])
        self.trainY, self.testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:len(dataY)])


    def MinMaxScaler(self, data):
        numerator = data - np.min(data, 0)
        denominator = np.max(data, 0) - np.min(data, 0)
        # noise term prevents the zero division
        return numerator / (denominator + 1e-7)

    def reverse_min_max_scaling(org_x, x):
        org_x_np = np.asarray(org_x)
        x_np = np.asarray(x)
        return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()

    def lstm_cell(self):
        cell = tf.contrib.rnn.LSTMCell(num_units=self.hidden_dim, state_is_tuple=True, reuse = tf.get_variable_scope().reuse)
        # cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1-self.drop_prob, 
        #                                  input_keep_prob=1-self.drop_prob)
        # cell = tf.contrib.rnn.GRUCell(num_units=self.hidden_dim)
        return cell

    def build_arch(self):
        with tf.device('/gpu:0'):
            with tf.variable_scope('Conv_layer'):
                conv1 = tf.layers.conv2d(inputs=self.X, filters=64, kernel_size=[3, 3], strides=(1,1), padding='same', name='conv1', reuse=tf.AUTO_REUSE)
                conv1 = tf.nn.relu(conv1, name='conv1_relu')
                assert conv1.get_shape() == [conv1.get_shape()[0], 9, 9, 64]

                conv2 = tf.layers.conv2d(inputs=conv1, filters=256, kernel_size=[3, 3], strides=(1,1), padding='same', name='conv2', reuse=tf.AUTO_REUSE)
                conv2 = tf.nn.relu(conv2, name='conv2_relu')
                assert conv2.get_shape() == [conv2.get_shape()[0], 9, 9, 256]

                # assert convt.get_shape() == [convt.get_shape()[0], 5, 5, 256]

                conv3 = tf.layers.conv2d(inputs=conv2, filters=16, kernel_size=[3, 3], strides=(1,1), padding='same', name='conv3', reuse=tf.AUTO_REUSE)
                conv3 = tf.nn.relu(conv3, name='conv3_relu')  
                assert conv3.get_shape() == [conv3.get_shape()[0], 9, 9, 16]

                conv4 = tf.layers.conv2d(inputs=conv3, filters=1, kernel_size=[1, 1], strides=(1,1), padding='valid', name='conv4', reuse=tf.AUTO_REUSE)
                conv4 = tf.nn.relu(conv4, name='conv4_relu')
                assert conv4.get_shape() == [conv4.get_shape()[0], 9, 9, 1]

                lstm_input = tf.reduce_sum(conv4, axis=3)
                assert lstm_input.get_shape() == [lstm_input.get_shape()[0], 9, 9]

            with tf.variable_scope('RNN_layer'):
                stackedRNNs = [self.lstm_cell() for _ in range(self.num_stacked_layers)]
                multi_cells = tf.contrib.rnn.MultiRNNCell(stackedRNNs, state_is_tuple=True)  if self.num_stacked_layers > 1 else self.lstm_cell()
         
                self.lstm_output, _states = tf.nn.dynamic_rnn(multi_cells, lstm_input, dtype=tf.float32)
            # print ('lstm_output', self.lstm_output.shape)
            # print ('shape', self.lstm_output[:, self.output_dim].shape)
            with tf.variable_scope('FullyConnected_layer'):
                fc1 = tf.contrib.layers.fully_connected(self.lstm_output[:, -1], self.fn_output1, activation_fn=None)
                fc2 = tf.contrib.layers.fully_connected(fc1, self.fn_output2, activation_fn=None)
                self.Y_pred = tf.contrib.layers.fully_connected(fc2, self.output_dim, activation_fn=None)

            # self.Y_pred = tf.contrib.layers.fully_connected(self.lstm_output[:, self.output_dim], self.output_dim, activation_fn=None)

    def loss(self):
        self.Euclidian_loss = tf.reduce_sum(tf.square(self.Y_pred - self.Y))  # sum of the squares
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train = self.optimizer.minimize(self.Euclidian_loss)

    def check(self):
        self.targets = tf.placeholder(tf.float32, [None, self.output_dim])
        self.predictions = tf.placeholder(tf.float32, [None, self.output_dim])
        self.rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(self.targets, self.predictions)))

    def run(self):
        self.DataLoading()
        self.build_arch()
        self.loss()
        self.check()

#         arr_conv = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
#         new_added_layer = ['conv_a1','conv_a2']
#         reuse_vars_dict = dict([(var.op.name, var) for var in arr_conv 
#                             if (new_added_layer[0] not in var.op.name) 
#                             and (new_added_layer[1] not in var.op.name) ])
#         restore_saver = tf.train.Saver(reuse_vars_dict)

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            self.savor = tf.train.Saver(max_to_keep=10)

#             restore_saver.restore(sess, 'drive/app/saved_v6-2/test_model-3000')
 

            for epoch in range(self.epoch_num):
                _, step_loss = sess.run([self.train, self.Euclidian_loss], feed_dict={self.X: self.trainX, self.Y: self.trainY})
                # print("[step: {}] loss: {}".format(i, step_loss))
                if ((epoch + 1) % 100 == 0) or (epoch == self.epoch_num - 1):  # every 100 step or last step
                    if (epoch + 1) % 1000 == 0 :
                        self.savor.save(sess, './saved_v6-a/test_model', global_step=epoch+1)
                    # get RMSE with train data
                    train_predict = sess.run(self.Y_pred, feed_dict={self.X: self.trainX})
                    train_error = sess.run(self.rmse, feed_dict={self.targets: self.trainY, self.predictions: train_predict})
                    self.train_error_summary.append(train_error)

                    # get RMSE with test data
                    test_predict = sess.run(self.Y_pred, feed_dict={self.X: self.testX})
                    test_error = sess.run(self.rmse, feed_dict={self.targets: self.testY, self.predictions: test_predict})
                    self.test_error_summary.append(test_error)
                    print("epoch: {}, train_error(A): {}, test_error(B): {}, B-A: {}".format(epoch+1, train_error, test_error, test_error - train_error))

            

            plt.figure(1)
            plt.plot(self.train_error_summary, 'gold')
            plt.plot(self.test_error_summary, 'b')
            plt.xlabel('Epoch(x100)')
            plt.ylabel('Root Mean Square Error')

            plt.figure(2)
            plt.plot(self.testY, 'r')
            plt.plot(test_predict, 'b')
            plt.xlabel('Time Period')
            plt.ylabel('Stock Price')
            plt.show()


if __name__ == "__main__":
    base_Learning = Learning()
    base_Learning.run()
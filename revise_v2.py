import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv

tf.set_random_seed(777)  # reproducibility


class Learning:
    def __init__(self):
        # AccPrice AccVolume open low high close ma boll rsi
        self.seq_length = 15
        self.data_dim = 9
        self.hidden_dim = 10
        self.output_dim = 1
        self.learning_rate = 0.001
        self.epoch_num = 3000
        self.num_stacked_layers = 2

        self.trainX = []
        self.trainY = []
        self.textX = []
        self.textY = []

        self.train_error_summary = []
        self.test_error_summary = []
        self.test_predict = ''  # prediction with test data

        self.DataLoading()

        self.X = tf.placeholder(tf.float32, [None, self.seq_length, self.data_dim])
        self.Y = tf.placeholder(tf.float32, [None, self.output_dim])

    def DataLoading(self):
        data = []
        with open('data_revise.csv', newline="") as csvfile:
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
            print(_x, "->", _y)
            dataX.append(_x)
            dataY.append(_y)

        train_size = int(len(dataY) * 0.7)
        test_size = len(dataY) - train_size
        self.trainX, self.testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
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
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_dim, state_is_tuple=True)
        # cell = tf.contrib.rnn.GRUCell(num_units=self.hidden_dim)
        return cell

    def build_arch(self):
        stackedRNNs = [self.lstm_cell() for _ in range(self.num_stacked_layers)]
        multi_cells = tf.contrib.rnn.MultiRNNCell(stackedRNNs, state_is_tuple=True) if self.num_stacked_layers > 1 else self.lstm_cell()
        self.lstm_output, _states = tf.nn.dynamic_rnn(multi_cells, self.X, dtype=tf.float32)
        self.Y_pred = tf.contrib.layers.fully_connected(self.lstm_output[:, self.output_dim], self.output_dim, activation_fn=None)

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

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            self.savor = tf.train.Saver()

            ckpt_state = tf.train.get_checkpoint_state("saved")
            # Training step
            ckpt_state.model_checkpoint_path
            for epoch in range(self.epoch_num):
                _, step_loss = sess.run([self.train, self.Euclidian_loss], feed_dict={self.X: self.trainX, self.Y: self.trainY})
                # print("[step: {}] loss: {}".format(i, step_loss))
                if ((epoch + 1) % 100 == 0) or (epoch == self.epoch_num - 1):  # every 100 step or last step
                    self.savor.save(sess, 'saved/test_model', global_step=100)
                    # get RMSE with train data
                    train_predict = sess.run(self.Y_pred, feed_dict={self.X: self.trainX})
                    train_error = sess.run(self.rmse, feed_dict={self.targets: self.trainY, self.predictions: train_predict})
                    self.train_error_summary.append(train_error)

                    # get RMSE with train data
                    test_predict = sess.run(self.Y_pred, feed_dict={self.X: self.testX})
                    test_error = sess.run(self.rmse, feed_dict={self.targets: self.testY, self.predictions: test_predict})
                    self.test_error_summary.append(test_error)
                    print("epoch: {}, train_error(A): {}, test_error(B): {}, B-A: {}".format(epoch+1, train_error, test_error, test_error - train_error))

            # test_predict = sess.run(self.Y_pred, feed_dict={self.X: self.testX})
            # rmse_val = sess.run(self.rmse, feed_dict={self.targets: self.testY, self.predictions: test_predict})

            # print("RMSE: {}".format(rmse_val))
            # Plot predictions
            # print(self.testY)
            # print(test_predict)
            # plt.plot(self.testY, label='actual')
            # plt.plot(test_predict, label='predict')
            # plt.xlabel("Time Period")
            # plt.ylabel("Stock Price")
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


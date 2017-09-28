import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

from mltest import GlycemiqDataContext as gcd

context = gcd.GlycemiqDataContext()

# Params
feature_count = len(context.get_data_columns())
label_count = len(context.get_label_columns())
batch_size = 6  # each row is a 5 minute interval; 6 * 5 = 30 min
learning_rate = 0.001
epochs = 800
display_epoch = 100
hidden_layer_neurons = 512
last_layer_neurons = 128


def create():
    # set weights and biases to random normal distribution
    weights = tf.Variable(tf.random_normal([last_layer_neurons, feature_count]), name="weights")
    biases = tf.Variable(tf.random_normal([label_count]), name="biases")

    # define input data and labels
    x_input = tf.placeholder(tf.float32, [batch_size, feature_count], name="x_input")
    y_input = tf.placeholder(tf.float32, [label_count], name="y_input")

    # reshape the inputs into batch sizes
    x_shaped = tf.reshape(x_input, [batch_size, feature_count], name="x_shaped")
    x_batched = tf.split(x_shaped, num_or_size_splits=batch_size, axis=0, name="x_batched")

    # declare the rnn
    rnn_cells = rnn.MultiRNNCell([rnn.BasicLSTMCell(hidden_layer_neurons),  # , activation=tf.nn.relu),
                                  rnn.BasicLSTMCell(hidden_layer_neurons),
                                  rnn.BasicLSTMCell(last_layer_neurons)])

    outputs, states = rnn.static_rnn(rnn_cells, x_batched, dtype=tf.float32)

    # not sure about just using the last output here
    prediction = tf.add(tf.matmul(outputs[-1], weights), biases, name="prediction")

    # loss optimization
    cost = tf.reduce_mean(tf.square(y_input - prediction), name="cost")
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, name="optimizer").minimize(cost)

    init = tf.global_variables_initializer()

    return init, optimizer, cost, prediction, x_input, y_input


def train(init, optimizer, cost, prediction, x_input, y_input):
    with tf.Session() as sess:
        sess.run(init)

        step = 0
        acc_total = 0.0
        cost_total = 0.0

        out_data = pd.DataFrame(columns=['step', 'prediction', 'actual', 'diff'])

        while step < epochs:
            # get data
            feature_data, label_data = context.get_next_training_data(batch_size)

            x_data = np.reshape(feature_data, [batch_size, feature_count])
            y_data = np.reshape(label_data, [1, label_count])

            _, cost_out, prediction_out = sess.run([optimizer, cost, prediction],
                                                   feed_dict={x_input: x_data, y_input: y_data[0]})

            cost_total += cost_out
            average_diff = np.mean(np.abs(y_data[0] - prediction_out[0]))
            acc_total += 1 - min([average_diff, 1])
            out_data.loc[step] = [step, prediction_out[0][0], y_data[0][0], average_diff]

            if ((step + 1) % display_epoch == 0) or step == 0:
                print("Iteration= %s" % str(1 + step))
                print("  Average Loss= {:1.4f}".format(cost_total / display_epoch))
                print("  Average Diff= {:1.4f}".format(average_diff))
                print("  Average Accuracy= {:3.2f}".format(100 * acc_total / display_epoch))
                print("  Actual - Predicted {:1.4f} vs {:1.4f} ".format(y_data[0][0], prediction_out[0][0]))

            step += 1

    out_data.to_csv(path_or_buf='C:/Users/Greg/Downloads/training_predictions.csv')


def test(init, cost, prediction, x_input, y_input):
    with tf.Session() as sess:
        sess.run(init)

        step = 0
        acc_total = 0.0
        cost_total = 0.0

        out_data = pd.DataFrame(columns=['step', 'prediction', 'actual', 'diff'])

        while step < epochs:
            # get data
            feature_data, label_data = context.get_next_test_data(batch_size)

            if feature_data is None:
                print('--- Test Complete ---')
                break

            x_data = np.reshape(feature_data, [batch_size, feature_count])
            y_data = np.reshape(label_data, [1, label_count])

            cost_out, prediction_out = sess.run([cost, prediction],
                                                feed_dict={x_input: x_data, y_input: y_data[0]})

            cost_total += cost_out
            average_diff = np.mean(np.abs(y_data[0] - prediction_out[0]))
            acc_total += 1 - min([average_diff, 1])
            out_data.loc[step] = [step, prediction_out[0][0], y_data[0][0], average_diff]

            if ((step + 1) % display_epoch == 0) or step == 0:
                print("Iteration= %s" % str(1 + step))
                print("  Average Loss= {:1.4f}".format(cost_total / display_epoch))
                print("  Average Diff= {:1.4f}".format(average_diff))
                print("  Average Accuracy= {:3.2f}".format(100 * acc_total / display_epoch))
                print("  Actual - Predicted {:1.4f} vs {:1.4f} ".format(y_data[0][0], prediction_out[0][0]))

            step += 1

    out_data.to_csv(path_or_buf='C:/Users/Greg/Downloads/test_predictions.csv')
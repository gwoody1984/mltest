import os
import pandas as pd
import numpy as np
import tensorflow as tf

from datetime import datetime
from tensorflow.contrib import rnn


class RnnModel:
    def __init__(self, context, batch_size=6, learning_rate=0.001, epochs=500, display_epoch=100,
                 hidden_layer_neurons=256, last_layer_neurons=128):
        self._context = context

        # File Params
        self._curr_time = datetime.now().strftime('%Y%m%d%H%M%S')
        cwd = os.getcwd()
        self._prediction_dir = cwd + "/prediction/"
        self._model_dir = cwd + "/model/"

        print('Current working directory set to {}'.format(cwd))

        if not os.path.exists(self._prediction_dir):
            os.makedirs(self._prediction_dir)
        if not os.path.exists(self._model_dir):
            os.makedirs(self._model_dir)

        # Model Params
        self._feature_count = len(self._context.get_data_columns())
        self._label_count = len(self._context.get_label_columns())
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._display_epoch = display_epoch
        self._hidden_layer_neurons = hidden_layer_neurons
        self._last_layer_neurons = last_layer_neurons

    def get_prediction_filename(self, prefix):
        return '{}{}_{}_prediction.csv'.format(self._prediction_dir, self._curr_time, prefix)

    def get_model_filename(self, steps):
        return '{}{}_model_{}'.format(self._model_dir, self._curr_time, str(steps))

    def create(self):
        # set weights and biases to random normal distribution
        weights = tf.Variable(tf.random_normal([self._last_layer_neurons, self._feature_count]), name="weights")
        biases = tf.Variable(tf.random_normal([self._label_count]), name="biases")

        # define input data and labels
        x_input = tf.placeholder(tf.float32, [self._batch_size, self._feature_count], name="x_input")
        y_input = tf.placeholder(tf.float32, [self._label_count], name="y_input")

        # reshape the inputs into batch sizes
        x_shaped = tf.reshape(x_input, [self._batch_size, self._feature_count], name="x_shaped")
        x_batched = tf.split(x_shaped, num_or_size_splits=self._batch_size, axis=0, name="x_batched")

        # declare the rnn
        rnn_cells = rnn.MultiRNNCell([rnn.BasicLSTMCell(self._hidden_layer_neurons),  # , activation=tf.nn.relu),
                                      rnn.BasicLSTMCell(self._hidden_layer_neurons),
                                      rnn.BasicLSTMCell(self._last_layer_neurons)])

        outputs, states = rnn.static_rnn(rnn_cells, x_batched, dtype=tf.float32)

        # not sure about just using the last output here
        prediction = tf.add(tf.matmul(outputs[-1], weights), biases, name="prediction")

        # loss optimization
        cost = tf.reduce_mean(tf.square(y_input - prediction), name="cost")
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self._learning_rate, name="optimizer").minimize(cost)

        init = tf.global_variables_initializer()

        return init, optimizer, cost, prediction, x_input, y_input

    def restore(self, model_file):
        if not os.path.exists(model_file):
            OSError('{} does not exist'.format(model_file))

        print('Restoring {}'.format(model_file))

        model_data_path = model_file[:-5]  # take off the .meta extension

        sess = tf.Session()

        saver = tf.train.import_meta_graph(model_file)
        saver.restore(sess, model_data_path)

        graph = tf.get_default_graph()
        x_input = graph.get_tensor_by_name("x_input:0")
        y_input = graph.get_tensor_by_name("y_input:0")
        prediction = graph.get_tensor_by_name("prediction:0")
        cost = graph.get_tensor_by_name("cost:0")

        return sess, graph, cost, prediction, x_input, y_input

    def train(self, init, optimizer, cost, prediction, x_input, y_input):
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)

            step = 0
            acc_total = 0.0
            cost_total = 0.0

            out_data = pd.DataFrame(columns=['step', 'batch_size', 'epochs', 'learning_rate',
                                             'hidden_layer_neurons', 'last_layer_neurons',
                                             'feature_count', 'prediction', 'actual', 'diff'])

            while step < self._epochs:
                # get data
                feature_data, label_data = self._context.get_next_training_data(self._batch_size)

                x_data = np.reshape(feature_data, [self._batch_size, self._feature_count])
                y_data = np.reshape(label_data, [1, self._label_count])

                _, cost_out, prediction_out = sess.run([optimizer, cost, prediction],
                                                       feed_dict={x_input: x_data, y_input: y_data[0]})

                cost_total += cost_out
                average_diff = np.mean(np.abs(y_data[0] - prediction_out[0]))
                acc_total += 1 - (average_diff / y_data[0])
                out_data.loc[step] = [step, self._batch_size, self._epochs, self._learning_rate,
                                      self._hidden_layer_neurons, self._last_layer_neurons, self._feature_count,
                                      prediction_out[0][0], y_data[0][0], average_diff]

                if (step + 1) % self._display_epoch == 0:
                    print("Iteration= %s" % str(1 + step))
                    print("  Average Loss= {:1.4f}".format(cost_total / (step + 1)))
                    print("  Current Diff= {:1.4f}".format(average_diff))
                    print("  Average Accuracy= {:1.2f}%".format(100 * acc_total[0] / (step + 1)))
                    print("  Actual - Predicted {:1.4f} vs {:1.4f} ".format(y_data[0][0], prediction_out[0][0]))

                step += 1

            # save the rnn
            save_path = saver.save(sess, self.get_model_filename(self._epochs)) + '.meta'

        prediction_csv_path = self.get_prediction_filename('train')
        out_data.to_csv(path_or_buf=self.get_prediction_filename('train'))

        print('Model saved to: {}'.format(save_path))
        print('Predictions saved to: {}'.format(prediction_csv_path))
        return save_path

    def test(self, model_file):
        sess, graph, cost, prediction, x_input, y_input = self.restore(model_file)
        print('Starting model test at {}'.format(datetime.now().strftime('%x %X')))

        with graph.as_default():
            with sess:

                step = 0
                acc_total = 0.0
                cost_total = 0.0

                out_data = pd.DataFrame(columns=['step', 'batch_size', 'epochs', 'learning_rate',
                                                 'hidden_layer_neurons', 'last_layer_neurons',
                                                 'feature_count', 'prediction', 'actual', 'diff'])

                while step < self._epochs:
                    # get data
                    feature_data, label_data = self._context.get_next_test_data(self._batch_size)

                    if feature_data is None:
                        print('--- Test Complete ---')
                        print("")
                        print("Final Iteration= %s" % str(1 + step))
                        print("  Average Loss= {:1.4f}".format(cost_total / self._display_epoch))
                        print("  Average Accuracy= {:1.2f}%".format(100 * acc_total[0] / (step + 1)))

                        break

                    x_data = np.reshape(feature_data, [self._batch_size, self._feature_count])
                    y_data = np.reshape(label_data, [1, self._label_count])

                    cost_out, prediction_out = sess.run([cost, prediction],
                                                        feed_dict={x_input: x_data, y_input: y_data[0]})

                    cost_total += cost_out
                    average_diff = np.mean(np.abs(y_data[0] - prediction_out[0]))
                    acc_total += 1 - (average_diff / y_data[0])
                    out_data.loc[step] = [step, self._batch_size, self._epochs, self._learning_rate,
                                          self._hidden_layer_neurons, self._last_layer_neurons, self._feature_count,
                                          prediction_out[0][0], y_data[0][0], average_diff]

                    if (step + 1) % self._display_epoch == 0:
                        print("Iteration= %s" % str(1 + step))
                        print("  Average Loss= {:1.4f}".format(cost_total / self._display_epoch))
                        print("  Current Diff= {:1.4f}".format(average_diff))
                        print("  Average Accuracy= {:1.2f}%".format(100 * acc_total[0] / (step + 1)))
                        print("  Actual - Predicted {:1.4f} vs {:1.4f} ".format(y_data[0][0], prediction_out[0][0]))

                    step += 1

            out_data.to_csv(path_or_buf=self.get_prediction_filename('test'))

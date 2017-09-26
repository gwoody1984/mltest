import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

from mltest import GlycemiqDataContext

context = GlycemiqDataContext()
test_data = context.get_data()

# feature engineering
test_data['diff5'] = np.log(test_data['bg']) - np.log(test_data['bg-5'])
test_data['diff10'] = np.log(test_data['bg-5']) - np.log(test_data['bg-10'])
test_data['diff15'] = (np.log(test_data['bg-10']) - np.log(test_data['bg-15']))
test_data['diff20'] = (np.log(test_data['bg-15']) - np.log(test_data['bg-20']))
test_data['diff25'] = (np.log(test_data['bg-20']) - np.log(test_data['bg-25']))
test_data['diff30'] = (np.log(test_data['bg-25']) - np.log(test_data['bg-30']))
#test_data['difflabel'] = np.log(test_data['label']) - np.log(test_data['bg'])

# list of columns we care about
cols = ['bg', 'diff5', 'diff10', 'diff15', 'diff20', 'diff25',
        'diff30', 'glycemicindex', 'calories', 'carbs', 'fiber', 'sugar',
        'basal_insulin', 'bolus_insulin']
labels = ['label']

# remove columns we don't need
x_data = test_data[cols]
y_data = test_data[labels]
ix = x_data.isnull().any(axis=1)
iy = y_data.isnull().any(axis=1)
x_data = test_data.loc[~ix, :]
y_data = test_data.loc[~iy, :]

# extract the test data from the labels
x_data = x_data.as_matrix(cols)
y_data = test_data.as_matrix(labels)

# Params
row_count, feature_count = x_data.shape
_, label_count = y_data.shape
batch_size = 6  # each row is a 5 minute interval; 6 * 5 = 30 min
learning_rate = 0.01
epochs = 5000
display_epoch = 100
hidden_layer_neurons = 32
last_layer_neurons = 16

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
rnn_cells = rnn.MultiRNNCell([rnn.BasicLSTMCell(hidden_layer_neurons, activation=tf.nn.relu),
                              rnn.BasicLSTMCell(hidden_layer_neurons, activation=tf.nn.relu),
                              rnn.BasicLSTMCell(last_layer_neurons)])

outputs, states = rnn.static_rnn(rnn_cells, x_batched, dtype=tf.float32)

# not sure about just using the last output here
prediction = tf.add(tf.matmul(outputs[-1], weights), biases, name="prediction")

# loss optimization
cost = tf.reduce_mean(tf.square(y_data - prediction), name="cost")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, name="optimizer").minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    step = 0
    acc_total = 0.0
    cost_total = 0.0

    while step < epochs:
        # get data
        x_matrix = np.reshape(x_data, [batch_size, feature_count])
        y_matrix = np.reshape(y_data, [1, label_count])

        _, cost_out, prediction_out = sess.run([optimizer, cost, prediction],
                                               feed_dict={x_input: x_matrix, y_input: y_matrix[0]})

        cost_total += cost_out
        average_diff = np.mean(np.abs(y_input[0] - prediction_out[0]))
        acc_total += 1 - min([average_diff, 1])

        if ((step + 1) % display_epoch == 0) or step == 0:
            print("Iteration= %s" % str(1 + step))
            print("  Average Loss= {:1.4f}".format(cost_total / step))
            print("  Average Accuracy= {:3.2f}".format(100 * acc_total / step))

        step += 1

from mltest import rnn_model as model, glycemiq_data_context as gcd

context = gcd.GlycemiqDataContext("5WG9G5")

rnn = model.RnnModel(context)
init, optimizer, cost, prediction, x_input, y_input = rnn.create()
model_file = rnn.train(init, optimizer, cost, prediction, x_input, y_input)
rnn.test(model_file)

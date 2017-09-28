from mltest import model

init, optimizer, cost, prediction, x_input, y_input = model.create()
model_file = model.train(init, optimizer, cost, prediction, x_input, y_input)
model.test(model_file)

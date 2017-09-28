from mltest import model

init, optimizer, cost, prediction, x_input, y_input = model.create()
model.train(init, optimizer, cost, prediction, x_input, y_input)
model.test(init, cost, prediction, x_input, y_input)

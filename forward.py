import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

train_data = 
test_data = 


x_train = train_data[:, 1:] / 255.0
y_train = train_data[:, 0].reshape((-1, 1))
x_test = test_data[:, 1:] / 255.0
y_test = test_data[:, 0].reshape((-1, 1))

y_train = (y_train == 3).astype(int)
y_test = (y_test == 3).astype(int)


input_size = 
hidden_size = 64
output_size = 1

weights1 = np.random.rand(input_size, hidden_size)
weights2 = np.random.rand(hidden_size, output_size)

learning_rate = 0.1
num_epochs = 10

for epoch in range(num_epochs):
 
    hidden_layer_input = np.dot(x_train, weights1)
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights2)
    output_layer_output = sigmoid(output_layer_input)
    
   
    loss = np.mean((output_layer_output - y_train)**2)
    accuracy = np.mean((output_layer_output > 0.5) == y_train)
    
  
    output_layer_error = (output_layer_output - y_train) * sigmoid_derivative(output_layer_output)
    hidden_layer_error = np.dot(output_layer_error, weights2.T) * sigmoid_derivative(hidden_layer_output)
    
    
    weights2 -= learning_rate * np.dot(hidden_layer_output.T, output_layer_error)
    weights1 -= learning_rate * np.dot(x_train.T, hidden_layer_error)
    
   
    print(f"Epoch {epoch+1}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")

hidden_layer_input = np.dot(x_test, weights1)
hidden_layer_output = sigmoid(hidden_layer_input)
output_layer_input = np.dot(hidden_layer_output, weights2)
output_layer_output = sigmoid(output_layer_input)
test_loss = np.mean((output_layer_output - y_test)**2)
test_accuracy = np.mean((output_layer_output > 0.5) == y_test)
print(f"Test Loss = {test_loss:.4f}, Test Accuracy = {test_accuracy:.4f}")

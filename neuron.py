class Neuron:
    def __init__(self, input_size):
        self.weights = np.random.normal(0, 1, input_size)
        self.bias = 0
    
    def forward(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        output = 1 / (1 + np.exp(-weighted_sum))
        return output

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_layer = [Neuron(input_size) for i in range(hidden_size)]
        self.output_layer = [Neuron(hidden_size) for i in range(output_size)]
    
    def forward(self, inputs):
        hidden_outputs = [neuron.forward(inputs) for neuron in self.hidden_layer]
        output = [neuron.forward(hidden_outputs) for neuron in self.output_layer]
        return output

nn = NeuralNetwork(2, 3, 1)


input_data = np.array([0.5, 0.8])
output = nn.forward(input_data)

print(output)

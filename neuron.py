class Neuron:
    def __init__(self, input_size):
        self.weights = np.random.normal(0, 1, input_size)
        self.bias = 0
    
    def forward(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        output = 1 / (1 + np.exp(-weighted_sum))
        return output

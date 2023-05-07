import numpy as np
input size =2
hidden_size = 3
output_size =1

W1 = np.random.normal(0, 1, (input_size, hidden_size))
W2 = np.random.normal(0, 1, (hidden_size, output_size))

b1 = np.zeros((1, hidden_size))
b2 = np.zeros((1, output_size))

import numpy as np

class softmax:
    def __init__(self, input_lens,nodes):
        self.weights = np.random.randn(input_lens,nodes)/input_lens
        self.biases = np.zeros(nodes)

    def forward(self, input):
        input = input.flatten()
        input_len, nodes =self.weights.shape
        totals =np.dot(input,self.weights) + self.biases

        exp = np.exp(totals)
        return exp/np.sum(exp, axis=0)

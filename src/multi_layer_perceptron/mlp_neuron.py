import random

from src.network.neuron import Neuron


class MlpNeuron:
    
    def __init__(self, num_inputs: int) -> None:
        """
        self.w are the wights of the network
        self.b i sth ebiad of the network
        """
        self.w = [Neuron(random.uniform(-1,1)) for _ in range(num_inputs)]
        self.b = Neuron(random.uniform(-1,1))
        
    def __call__(self, n):
        """
        n: should be a valid int or float tensor or list
        -------
        The goal of this is to return w.n + b
        where w.n is the dot product between the weights and the inputs and b is the bias.
        """
        value = sum((wi * ni for wi, ni in zip(self.w, n)), self.b)
        activated_value = value.tanh()
        return activated_value
    
    
class Layer:
    """
    A layer is nothing but a set of neurons.
    In this case there is an element of homogenity wherein each neuron in a particular layer has the same number of inputs
    """
    def __init__(self, num_inputs, num_neurons) -> None:
        self.neurons = [MlpNeuron(num_inputs) for _ in range(num_neurons)]
        
    def __call__(self, x):
        outputs = [n(x) for n in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs
    
class Mlp:
    """
    A Multi Layer Perceptron class.
    """
    
    def __init__(self, num_inputs, num_neurons_per_layer: list) -> None:
        """
        Let's say there are 3 inputs, 2 layers with 4 neurons each and 1 output
        num_inputs would be 3
        num_neurons_per_layer will be [4, 4, 1] -> 4 neurons each hidden layer and a single output
        """
        sizer = [num_inputs] + num_neurons_per_layer
        self.layers = [Layer(sizer[i], sizer[i+1]) for i in range(len(num_neurons_per_layer))]
        
    def __call__(self, x):
        """
        This is equivalent to the forward pass.
        the returned value is that of the output neuron
        """
        for layer in self.layers:
            x = layer(x)
        return x
        
    
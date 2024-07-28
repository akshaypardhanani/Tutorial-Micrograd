"""
Copied this over from Perceptron.py
Difference is that I added the additional functionality that you see in the tutorials separately.
This is done purely for my own understanding and tracking the evolution of the explanation.
"""

import math
from src.network.value import Value

class Neuron(Value):
    """
    Override the __add__ and __mul__ methods here to allow for backpropagation programmatically.
    Accumulate the gradients in that case so as to allow for the multi-variate derivatives case.
    
    Add the option of the activation function to squach values between -1.0 and 1.0 (tanh)
    """
    def __init__(self, data, run=None, _children=(), _op='', label='', *args, **kwargs):
        _children = kwargs.get('_prev', ()) if len(_children) < 1 else _children
        super().__init__(data, run, _children, _op, label)
        
        self._backward = None
        
    def __repr__(self) -> str:
        return f"Value(data={self.data})"
        
    def __neg__(self):
        return self * -1
        
    def __add__(self, other):
        other = Neuron(other) if not (isinstance(other, Value) or isinstance(other, Neuron)) else other
        out = super().__add__(other)
        out = Neuron(**out.__dict__)
        def _distribute_gradient():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
            
        out._backward = _distribute_gradient
        
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        return self + (-other)
    
    def __mul__(self, other):
        other = Neuron(other) if not (isinstance(other, Value) or isinstance(other, Neuron)) else other
        out = super().__mul__(other)
        out = Neuron(**out.__dict__)
        def _combine_gradient():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
            
        out._backward = _combine_gradient
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __pow__(self, other): # self ^ other
        if not isinstance(other, (int, float)):
            raise ValueError("Only int and float powers are supported right now")
        out = Neuron(self.data ** other, _children=(self, ), _op=f"**{other}")
        
        def _gradient():
            self.grad += (other * (self.data ** (other - 1))) * out.grad
        out._backward = _gradient
        
        return out
    
    def __truediv__(self, other): # self / other
        return self * other ** -1
    
    def exp(self):
        x = self.data
        out = Neuron(math.exp(x), _children=(self, ), _op='exp')
        
        def _gradient():
            self.grad += out.data * out.grad # derivative wrt x of e^x = e^x which is `out`
        out._backward = _gradient
        
        return out
        
    @staticmethod
    def _build_topoligical_graph(node: Value, topo: list = [], visited: set = set()) -> list:
        if node not in visited:
            visited.add(node)
            for child in node._prev:
                Neuron._build_topoligical_graph(child, topo, visited)
            topo.append(node)
        return topo
        
    def tanh(self):
        """
        The value itself of tanh is given  by:
                      e^2x - 1
            tanh(x) = -----------
                      e^2x + 1
        The derivative of tanh is 1 - tanh^2(x)
        """
        
        x = self.data # activation is computed at the node
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Neuron(t, _children=(self, ), _op='tanh')
        
        def _derivative_tanh():
            # derivative of tanh time the gradient of the output
            self.grad += (1 - t**2) * out.grad
        out._backward = _derivative_tanh
        return out
        
    def backward(self):
        topo = Neuron._build_topoligical_graph(self)
        self.grad = 1.0
        for node in reversed(topo):
            if node._backward:
                node._backward()
    
    
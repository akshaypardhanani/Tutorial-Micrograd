"""
Value representation that also support expression graphs.
We want to know what values produced other values.
"""

class Value:
    """
    _children: Keep strack of the values that were produces by some operation
    _op: Is the operation that produced some value
    """
    def __init__(self, data, run = None, _children = (), _op='', label = '') -> None:
        self.data = data + run if run else data
        # This is the gradient of the function. 
        # In other words this value represents how much this value affects the output of the function and in what direction
        # In the forward pass this will always be zero and the values get computed during back propagation
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        
        # purely used for visual purposes has no functional impact
        self.label = label
        
        
    def __repr__(self) -> str:
        return f"Value(data={self.data}, children={self._children}, _operation={self._op})"
    
    def __add__(self, other):
        return Value(self.data + other.data, _children=(self, other), _op='+')
    
    def __mul__(self, other):
        return Value(self.data * other.data, _children=(self, other), _op='*')
    

from src.network.value import Value
from src.utils.draw import draw_dot

"""
The initial forward pass during which the values of the function for the inputs are calculated 
"""

def initial_values():
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a*b; e.label = 'e'
    d = e + c; d.label = 'd'
    f = Value(-2.0, label='f')
    L = d * f; L.label = 'L'
    return L
    
if __name__ == '__main__':
    dot = draw_dot(initial_values())
    dot.render('forward.svg', view=True)
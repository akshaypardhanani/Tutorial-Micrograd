"""
The derivative of the output `L` with respect to itself will also ways be 1.
Therefore we can directly set its gradient to 1.0
"""
from enum import Enum
from typing import Union
from src.network.value import Value
from src.utils.draw import draw_dot


class DeriveAt(Enum):
    A: str = 'a'
    B: str = 'b'
    C: str = 'c'
    D: str = 'd'
    E: str = 'e'
    F: str = 'f'
    L: str = 'l'
    
def derivative(der_of: Value, der_wrt: Value, h: float):
    return (der_of.data - der_wrt.data)/h
    
    
def network_function(a: float = 2.0, b: float = -3.0, c: float = 10.0, f: float = -2.0, h: float = 0.001, 
                     derive_at: Union[None, DeriveAt] = None):
    _a = Value(a + h, label='a') if derive_at and DeriveAt(derive_at).value == 'a' else Value(a, label='a')
    _b = Value(b + h, label='b') if derive_at and DeriveAt(derive_at).value == 'b' else Value(b, label='b')
    _c = Value(c + h, label='c') if derive_at and DeriveAt(derive_at).value == 'c' else Value(c, label='c')
    
    e = _a * _b 
    e.label = 'e'
    if derive_at and DeriveAt(derive_at).value == 'e':
        e.data += h
    
    d = e + _c 
    d.label = 'd'
    if derive_at and DeriveAt(derive_at).value == 'd':
        d.data += h
    
    _f = Value(f + h, label='f') if derive_at and DeriveAt(derive_at).value == 'f' else Value(f, label='f')
    L = d * _f
    L.label = 'L'
    if derive_at and DeriveAt(derive_at).value == 'l':
        L.data += h
    
    return L

def backpropagate(h: float = 0.001):
    l1 = network_function()
    l2 = network_function(derive_at='l')
    derivative_l = derivative(l2, l1, h)
    print("gradient at L", derivative_l)
    
    grad_f = network_function(derive_at='f')
    derivative_f = derivative(grad_f, l1, h)
    print("gradient at f", derivative_f)
    
    grad_d = network_function(derive_at='d')
    derivative_d = derivative(grad_d, l1, h)
    print("gradient at d", derivative_d)
    
    grad_e = network_function(derive_at='e')
    derivative_e = derivative(grad_e, l1, h)
    print("gradient at e", derivative_e)
    
    grad_c = network_function(derive_at='c')
    derivative_c = derivative(grad_c, l1, h)
    print("gradient at c", derivative_c)
    
    grad_b = network_function(derive_at='b')
    derivative_b = derivative(grad_b, l1, h)
    print("gradient at b", derivative_b)
    
    grad_a = network_function(derive_at='a')
    derivative_a = derivative(grad_a, l1, h)
    print("gradient at a", derivative_a)

    
if __name__ == '__main__':
    backpropagate()

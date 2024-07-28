"""
Evaluate how much the output of a function changes when the input is modified
"""
from typing import Union

# Define some scalars
a = 2.0
b = -3.0
c = 10

def func(a: Union[int, float], b: Union[int, float], c: Union[int, float]) -> Union[int, float]:
    return a * b + c

if __name__ == '__main__':
    d1 = func(a,b,c)
    h = 0.0001
    d2 = func(a + h, b, c)
    print("d1", d1)
    print("d2", d2)
    print("slope a", (d2 - d1)/h)
    
    db = func(a, b + h, c)
    print("slope b", (db - d1)/h)
    
    dc = func(a, b, c + h)
    print("slope c", (dc - d1)/h)
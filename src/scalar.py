"""
Scalar function that takes some input and gives a corresponding output.
Used to generate a mapping from x -> y
"""

from typing import Union

def f(x: Union[int, float]) -> Union[int, float]:
    return 3*x**2 - 4 * x +5

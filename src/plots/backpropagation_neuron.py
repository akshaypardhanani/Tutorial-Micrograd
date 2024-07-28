from src.network.neuron import Neuron
from src.utils.draw import draw_dot

def init_inputs():
    x1 = Neuron(2.0, label='x1')
    x2 = Neuron(0.0, label='x2')
    return x1, x2
    
def init_weights():
    w1 = Neuron(-3.0, label='w1')
    w2 = Neuron(1.0, label='w2')
    return w1, w2
    
def init_bias():
    return Neuron(6.8813735870195432, label='b')

def forward():
    x1, x2 = init_inputs()
    w1, w2 = init_weights()
    b = init_bias()
    
    x1_w1 = x1 * w1
    x1_w1.label = 'x1.w1'
    
    x2_w2 = x2 * w2
    x2_w2.label = 'x2.w2'
    
    x1_w1_x2_w2 = x1_w1 + x2_w2
    x1_w1_x2_w2.label = 'x1.w1 + x2.w2'
    
    n = x1_w1_x2_w2 + b
    n.label = 'n'
    
    # o = n.tanh() # call activation function
    # This is the same as calling tanh
    e = (2*n).exp()
    e.label = 'e'
    o = (e - 1) / (e + 1)
    o.label = 'o'
    
    # fig = draw_dot(o)
    # fig.render('forward_auto', view=True)
    
    return o
    
if __name__ == '__main__':
    network = forward()
    network.backward()
    
    fig = draw_dot(network)
    fig.render('backprop_auto', view=True)
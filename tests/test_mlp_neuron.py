from src.multi_layer_perceptron.mlp_neuron import Mlp
from src.utils.draw import draw_dot


def test_forward_pass_works():
    inputs = [2.0, 3.0, -1.0]
    network = Mlp(3, [4,4,1])
    result = network(inputs)
    fig = draw_dot(result)
    fig.render('mlp_forward', view=True)
    
    
def test_mlp_with_tensor():
    network = Mlp(3, [4,4,1])
    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0],]
    ys = [1.0, -1.0, -1.0, 1.0] # desired targets
    
    ypreds = [network(x) for x in xs]
    print(ypreds)
    
def test_with_loss():
    network = Mlp(3, [4,4,1])
    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0],]
    ys = [1.0, -1.0, -1.0, 1.0] # desired targets
    
    ypreds = [network(x) for x in xs]
    
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypreds))
    
    print(f"Loss is: {loss}")
    
    loss.backward()
    
    fig = draw_dot(loss)
    fig.render('mlp_after_loss', view=True)
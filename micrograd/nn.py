import random
from micrograd.engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    # nin = num of inputs into neuron
    # creates a weight and a bias
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin
    
    # magic method to allow instance of object to treated like function
    # n = Neuron(4) -> n(some vector)
    # zip() makes an iterator that iterates over entries that have been zipped into tuple
    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    # parameters are the weights and an adjustable part of the neuron 
    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    # layer is list of neurons
    # kwargs for arbitrary number of args
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    """
    Type: Multilayer Perceptron (MLP) is a basic neural network in machine learning.
    Structure: Comprises an input layer, one or more hidden layers, and an output layer.
    Connectivity: Fully connected; each neuron links to every neuron in the next layer.
    Activation Functions: Utilizes non-linear functions like sigmoid, tanh, or ReLU.
    Data Flow: Feedforward, with data moving in one direction from input to output.
    Training: Typically trained using backpropagation.
    Applications: Suitable for classification, regression, but less effective for spatial or sequential data.
    Limitations: Struggles with complex, large datasets and has inefficiencies with increasing input size.
    """ 

    # the connected layers
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

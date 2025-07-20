import random
from puregrad.utils import get_kaiming_gain
from puregrad.value import Value

class Module():
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []
    
class Neuron(Module):

    def __init__(self, n_in, nonlin="relu", lrelu_neg_slope=None):
        gain = get_kaiming_gain(nonlin, lrelu_neg_slope)

        nonlin_2_func = {
            "linear": lambda x: x,           
            "tanh": lambda x: x.tanh(),      
            "relu": lambda x: x.relu(),      
            "lrelu": lambda x: x.leaky_relu(0.01 if lrelu_neg_slope is None else lrelu_neg_slope)
        }

        self.non_lin = nonlin_2_func[nonlin]
        self.w = [Value(random.uniform(-1, 1)) * gain for _ in range(n_in)]
        
        # Adjust bias initialisation depending on nonlinearity
        if nonlin in ["relu", "lrelu"]:
            self.b = Value(0.01) 
        elif nonlin == "linear":
            self.b = Value(0.0) 
        else:
            self.b = Value(random.uniform(-0.1, 0.1))

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return self.non_lin(act)
    
    def parameters(self):
        return self.w + [self.b]


    
class Linear(Module):

    def __init__(self, n_in, n_out, non_lin="relu", lrelu_neg_slope=None):
        self.neurons = [Neuron(n_in, non_lin, lrelu_neg_slope) for _ in range(n_out)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    
class MLP(Module):

    def __init__(self, n_in, n_outs, non_lin="relu", lrelu_neg_slope=None):
        sz = [n_in] + n_outs
        self.layers = [Linear(sz[i], sz[i+1], non_lin=non_lin if i!=len(n_outs)-1 else "linear", lrelu_neg_slope=lrelu_neg_slope) for i in range(len(n_outs))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
        
        
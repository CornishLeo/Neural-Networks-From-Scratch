import random
from mygrad.value import Value

class Module():
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []
    
class Neuron(Module):

    def __init__(self, n_in, non_lin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0)
        self.non_lin = non_lin

    def __call__(self, x):
        wx = [wi*xi for wi, xi in zip(self.w, x)]
        activation = [y + self.b for y in wx]
        return activation.relu() if self.non_lin else activation
    
    def parameters(self):
        return self.w + [self.b]
    
class Layer(Module):

    def __init__(self, n_in, n_out, non_lin=True):
        self.neurons = [Neuron(n_in, non_lin) for _ in range(n_out)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters]
    
class MLP(Module):

    def __init__(self, n_in, n_outs):
        sz = [n_in] + n_outs
        self.layers = [Layer(sz[i], sz[i+1], non_lin=i!=len(n_outs)-1) for i in range(len(n_outs))]
        
        
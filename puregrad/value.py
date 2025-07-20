import math

class Value():

    def __init__(self, data, _children=(), _op=''):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def backward(self):

        topo = []
        seen = set()

        def build_topo(v):
            if v not in seen:
                seen.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def tanh(self):
        out = Value(math.tanh(self.data), (self,), 'tanh')

        def _backward():
            self.grad += (1 - out.data**2.0) * out.grad
        out._backward = _backward

        return out
    
    def relu(self):
        out = Value(0.0 if self.data < 0 else self.data, (self,), 'relu')

        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward

        return out
    
    def leaky_relu(self, alpha=0.01):  # alpha is the slope for x < 0
        out = Value(max(alpha * self.data, self.data), (self,), 'l_relu')

        def _backward():
            self.grad += (1.0 if self.data > 0 else alpha) * out.grad

        out._backward = _backward
        return out

    
    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def log(self):
        epsilon = 1e-8  # Small constant
        safe_data = max(self.data, epsilon)  # Avoid log(0) error
        out = Value(math.log(safe_data), (self,), 'log')

        def _backward():
            self.grad += (1.0 / safe_data) * out.grad
        out._backward = _backward

        return out

    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Value(other)
        elif isinstance(other, Value):
            pass
        else:
            raise Exception("Sorry, data type not supported for adding with Value object")
        
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
 
        return out
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = Value(other)
        elif isinstance(other, Value):
            pass
        else:
            raise Exception("Sorry, data type not supported for multiplying with Value object")
            
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
 
        return out
    
    def __pow__(self, other):
        if isinstance(other, (int, float)):
            other = Value(other)
        elif isinstance(other, Value):
            pass
        else:
            raise Exception("Sorry, data type not supported for Value to the power of")
            
        out = Value(self.data**other.data, (self, other), "**")

        def _backward():
            self.grad += (other.data * self.data**(other.data - 1.0)) * out.grad

        out._backward = _backward
 
        return out
    
    def __radd__(self, other):
        return self.__add__(Value(other))
    
    def __neg__(self):
        return self * Value(-1.0)
    
    def __sub__(self, other):          
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1.0

    def __rtruediv__(self, other):
        return other * self**-1.0
    
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


    
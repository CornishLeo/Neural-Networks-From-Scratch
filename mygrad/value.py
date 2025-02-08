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
            self.grad += (1 - out.data**2) * out.grad
        out._backward = _backward

        return out
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'relu')

        def _backward():
            self.grad += (self.data > 0) * out.grad
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
            self.grad += other.data * self.data**(other.data - 1) * out.grad

        out._backward = _backward
 
        return out
    
    def __radd__(self, other):
        return self.__add__(Value(other))
    
    def __neg__(self):
        return self * Value(-1.0)
    
    def __sub__(self, other):          
        return self.data + (-other)
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
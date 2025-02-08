# A class to represent basic functionality of backpropagation
class Value():

    def __init__(self, data, _children=(), opp=None):
        self.data = data
        self._prev = set(_children)
        self.opp = opp

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Value(other)
        elif isinstance(other, Value):
            pass
        else:
            raise Exception("Sorry, data type not supported for adding with Value object")
            
        return Value(self.data + other.data, (self, other), "+")
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, (int, float)):
            other = Value(other)
        elif isinstance(other, Value):
            pass
        else:
            raise Exception("Sorry, data type not supported for subtracting with Value object")
            
        return Value(self.data - other.data, (self, other), "-")
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = Value(other)
        elif isinstance(other, Value):
            pass
        else:
            raise Exception("Sorry, data type not supported for multiplying with Value object")
            
        return Value(self.data * other.data, (self, other), "*")
    
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            other = Value(other)
        elif isinstance(other, Value):
            pass
        else:
            raise Exception("Sorry, data type not supported for true division with Value object")
            
        return Value(self.data / other.data, (self, other), "/")
    
    def __pow__(self, other):
        if isinstance(other, (int, float)):
            other = Value(other)
        elif isinstance(other, Value):
            pass
        else:
            raise Exception("Sorry, data type not supported for Value to the power of")
            
        return Value(self.data**other.data, (self, other), "**")
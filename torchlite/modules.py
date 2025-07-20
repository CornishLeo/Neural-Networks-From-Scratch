import torch

class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}

    def parameters(self):
        params = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

    def add_parameter(self, name, param):
        self._parameters[name] = param

    def add_module(self, name, module):
        self._modules[name] = module

    def forward(self, *inputs):
        raise NotImplementedError

    def __call__(self, *inputs):
        return self.forward(*inputs)

    def zero_grad(self):
        for param in self.parameters():
            if param.grad is not None:
                param.grad.zero_()


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        self.weight = torch.randn(out_features, in_features, requires_grad=True) * (2 / in_features) ** 0.5
        self.add_parameter("weight", self.weight)

        if bias:
            self.bias = torch.zeros((out_features))
            self.add_parameter("bias", self.self.bias)
        else:
            self.bias = None

    def forward(self, x):
        y = x @ self.weight.t()
        if self.bias is not None:
            y += self.bias
        return y


class ReLU(Module):
    def forward(self, x):
        return x.clamp(min=0)

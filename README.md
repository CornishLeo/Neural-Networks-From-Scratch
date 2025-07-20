# Neural-Networks-From-Scratch

This project is an educational deep dive into building neural networks from scratch, inspired by PyTorch and the teachings of Andrej Karpathy. The aim is to reinforce an intuitive and mathematical understanding of deep learning fundamentals, including backpropagation and gradient descent.

The project currently includes two main components:

## puregrad: Pure Python Neural Network Framework

A minimalist framework written in pure Python, with no external libraries except for `graphviz` (used for visualization only).

### What's inside `puregrad/`:
- `value.py` — A scalar `Value` class implementing reverse-mode autodiff (like micrograd).
- `nn.py` — A simple neural network framework using `Value` for backpropagation.
- `utils.py` — Utility functions such as activation functions, loss computations, and initializations.
- `scripts/` — Training and evaluation scripts, e.g.:
  - `iris_classifier.py` — A working example of training an MLP on the Iris dataset (download link inside).
  - `MNIST_classifier.py` — A working example of training an MLP on the Iris dataset (download link inside).
- `models/` — Directory to save and load trained models (via `pickle` or similar).
- `example_value.ipynb` — A Jupyter notebook that visualizes the computation graph using `graphviz`.

### How to run an example:

From the project root, run the following command to train a model on the Iris dataset:
```bash
python -m puregrad.scripts.iris_classifier
```

Note: `graphviz` is only used inside `example_value.ipynb` to show how the computation graph is built and traversed — it's not needed for training or running models.

## torchlite: PyTorch-Based Reimplementation (in progress)

In addition to the pure Python version, this project includes an alternate implementation called `torchlite` — a minimal neural network framework that leverages PyTorch tensors and autograd, but rebuilds the rest (e.g., layers, loss functions, optimizers) from scratch.

### What's planned/included in `torchlite/`:
- Custom `Module`, `Linear`, and activation classes built on top of `torch.Tensor`.
- Manual training loops using `.backward()` and `.zero_grad()` for educational clarity.
- Cleaner, GPU-compatible code with PyTorch's performance benefits.

This part of the project is necessary to implement larger and more complex models without waiting years for it to run the grad engine i wrote on the cpu...

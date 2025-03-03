# Neural-Networks-From-Scratch

A project to implement neural networks from scratch.

This project aims to deliver a functional AI framework loosely based on PyTorch and inspired by the teachings of Andrej Karpathy. The goal is to deepen my understanding of the foundational mathematics behind neural networks, such as gradient descent.

Currently this code uses pure python code with no external libraries

## How to use

Currently only linear models are implemented with more coming in the future, you can train this by importing the MLP Class from mygrad/nn.py , this implements a Multilayer Perceptron using the mygrad/value.py for creating a graph to backpropagate through.

The iris_classifier.py uses the is an example of how to set up a simple MLP on a common simple training data of predicting classes of iris flowers

The value_example.ipynb uses graphviz to demonstrate how the Value class builds graph that gradients can be determined through 

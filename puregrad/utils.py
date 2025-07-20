import json
from puregrad.value import Value

def one_hot(idx, num_classes):
    encoding = [0 for _ in range(num_classes)]
    encoding[idx] = 1
    return encoding

def cross_entropy(logits, target):
    # subtract the max for numerical stability
    max_val = max_value(logits)
    logits = [(val - max_val) for val in logits]  # Ensure values remain linked


    # 1) evaluate elementwise e^x
    ex = [x.exp() for x in logits]
    # 2) compute the sum of the above
    denom = sum(ex, Value(0.0))
    # 3) normalize by the sum to get probabilities
    probs = [x / denom for x in ex]
    # 4) log the probabilities at target
    logp = (probs[target]).log()
    # 5) the negative log likelihood loss (invert so we get a loss - lower is better)
    nll = -logp
    return nll

# Returns the max value of a list
def max_value(val_list):

    if len(val_list) == 0:
        raise Exception("Cannot find maximum of empty list")

    max_value = val_list[0]
    for val in val_list[1:]:
        if val.data > max_value.data:
            max_value = val

    return max_value

# Returns the index of the max value of a list
def max_index(val_list):
    max_value = val_list[0]
    max_idx = 0
    for idx, val in enumerate(val_list):
        if val.data > max_value.data:
            max_value = val
            max_idx = idx
    return max_idx  # Return correct index

def get_kaiming_gain(nonlin, slope=0.01):

    if slope is None:
        slope = 0.01

    nonlin_2_gain = {
        "linear": 1,
        "sigmoid": 1,
        "tanh": 5/3,
        "relu": 2**0.5,
        "lrelu": (2 / (1 + slope**2))**0.5
    }

    return nonlin_2_gain[nonlin]

def save_model(model_object, file_output):
    if not file_output.endswith(".json"):  # Save as JSON for better readability
        file_output += ".json"

    if model_object.__class__.__name__ == "MLP":
        model_data = {
            "model_type": "MLP",
            "layer_sizes": [len(layer.neurons) for layer in model_object.layers],
            "weights": [[neuron.w[i].data for i in range(len(neuron.w))] for layer in model_object.layers for neuron in layer.neurons],
            "biases": [[neuron.b.data for neuron in layer.neurons] for layer in model_object.layers]
        }

        with open(file_output, "w") as file:
            json.dump(model_data, file, indent=4)  # Save as JSON

        print(f"Model saved to {file_output}")
    else:
        print("The system does not support saving this object")

def load_model(model_path):
    from puregrad.nn import MLP

    with open(model_path, "r") as file:
        model_data = json.load(file)

    if model_data["model_type"] != "MLP":
        raise ValueError("Invalid model type in file")
    
    layer_sizes = model_data["layer_sizes"]
    weights_data = model_data["weights"]
    biases_data = model_data["biases"]

    # Determine the input size from the first weight vector
    input_size = len(weights_data[0])
    model = MLP(input_size, layer_sizes)

    weight_index = 0
    # Iterate over layers (each corresponding to a group of neurons)
    for layer_idx, layer in enumerate(model.layers):
        biases_for_layer = biases_data[layer_idx]  # This list contains biases for each neuron in the layer
        for neuron_idx, neuron in enumerate(layer.neurons):
            neuron.w = [Value(w) for w in weights_data[weight_index]]
            neuron.b = Value(biases_for_layer[neuron_idx])
            weight_index += 1

    print(f"Model loaded from {model_path}")
    return model

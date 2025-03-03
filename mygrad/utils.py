from graphviz import Digraph

from mygrad.value import Value


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

# Function for getting all nodes and edges to draw nn graph
def trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)

        for child in v._prev:
            edges.add((child, v))
            build(child)

    build(root)
    return nodes, edges

# Draws the graph using graphviz
def draw_dot(root):
    
    dot = Digraph(format="svg", graph_attr={'rankdir': 'LR'})

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))

        dot.node(name=uid, label= "{ data: %.4f \n grad: %.4f}" % (n.data, n.grad), shape='record')
        if n._op:
            dot.node(name=uid + n._op, label = n._op)
            dot.edge(uid + n._op, uid)
        
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

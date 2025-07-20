import csv
import random
from puregrad.nn import MLP
from puregrad.utils import cross_entropy, max_index, save_model

# Hyperparameters
train_split = 0.6
lr = 0.01
epochs = 200
hidden_sizes = [10, 5] 

# Get data from csv data (https://gist.github.com/netj/8836201)
with open('datasets/iris.csv', 'r') as read_obj: 
    csv_reader = csv.reader(read_obj) 
    list_of_csv = list(csv_reader) 
  
# Collect data and convert to floats when appropriate
headers = list_of_csv[0]
data = []
for row in list_of_csv[1:]:
    temp_row = []
    for item in row[:-1]:
        temp_row.append(float(item))

    temp_row.append(row[-1])
    data.append(temp_row)

# Shuffle for sampling
random.shuffle(data)
len_data = len(data)
train_size = round(len_data * train_split)

# Split features and result
X = [row[:-1] for row in data]
y = [row[-1] for row in data]

# Collect class types and create conversion dicts
classes = sorted(list(set(y)))
class_size = len(classes)
class_2_int = {}
int_2_class = {}

for idx, flower in enumerate(classes):
    class_2_int[flower] = idx
    int_2_class[idx] = flower
print("class to int")
print(class_2_int)
# Convert y to int values based on classes
y = [class_2_int[a] for a in y]

# Collect train test splits
Xtr, ytr = X[:train_size], y[:train_size]
xtest, ytest = X[train_size:], y[train_size:]

# Create MLP
model = MLP(len(Xtr[0]), hidden_sizes + [len(classes)])

print(f"Number of classes: {class_size}")

for k in range(epochs):

    # Reset gradients
    model.zero_grad()

    # Get predicitons for training examples
    scores = list(map(model, Xtr))

    losses = list(map(cross_entropy, scores, ytr))

    avg_loss = sum(losses) / len(losses)
    avg_loss.backward()

    for p in model.parameters():
        p.data -= lr * p.grad 


    print(f"Epoch {k+1}, Loss: {avg_loss.data}")

    
    # Calculate accuracy for the epoch
    test_score_idx = list(map(max_index, scores))

    count = 0
    true_count = 0
    for idx, score_idx in enumerate(test_score_idx):
        count += 1
        if score_idx == ytr[idx]:
            true_count += 1
    print(f"accuracy: {true_count / count}")



# Calculate accuracy with test data
test_scores = list(map(model, xtest))
test_score_idx = list(map(max_index, test_scores))

count = 0
true_count = 0
for idx, score_idx in enumerate(test_score_idx):
    count += 1
    if score_idx == ytest[idx]:
        true_count += 1
print(f"accuracy: {true_count / count}")

save_model(model, "puregrad/models/iris_classifier")


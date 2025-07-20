import csv
import random

from puregrad.nn import MLP
from puregrad.utils import cross_entropy, load_model, max_index, save_model

# Open the file (downloaded from https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
# and create Xtr and ytr
with open("datasets/mnist_train.csv", mode="r", encoding="utf-8") as file:
    reader = csv.reader(file) 
    Xtr = []
    ytr = []
    header = True
    for row in reader:
        if header:
            header = False
        else:
            Xtr.append([int(i) / 255.0 for i in row[1:]])
            ytr.append(int(row[0]))

# Get number of classes
classes = list(set(ytr))
len_data = len(ytr)
print(f"Length of data: {len_data}")

# Hyperparameters
epochs = 100
hidden_sizes = [80, 64]
batch_size = 60

model = MLP(len(Xtr[0]), hidden_sizes + [len(classes)])
# model = load_model("MNIST_Classifier_1.json")

print("Started training...\n")

for k in range(epochs):

    print(f"epoch: {k+1}")

    # Reset gradients
    model.zero_grad()

    batch_X = []
    batch_y = []
    # Build batch
    for i in range(batch_size):
        temp_idx = random.randint(0, len_data -1)
        batch_X.append(Xtr[temp_idx])
        batch_y.append(ytr[temp_idx])

    # Get predicitons for training examples
    scores = [model(x) for x in batch_X]


    losses = list(map(cross_entropy, scores, batch_y))

    avg_loss = sum(losses) / len(losses)
    avg_loss.backward()

    for p in model.parameters():
        p.data -= p.grad * 0.1 if k < (0.8 * epochs) else 0.01 # Decay learning rate after 80% of training

    print(f"Loss: {avg_loss.data}")
    
    # Calculate accuracy for the epoch
    test_score_idx = list(map(max_index, scores))

    count = 0
    true_count = 0
    for idx, score_idx in enumerate(test_score_idx):
        count += 1
        if score_idx == batch_y[idx]:
            true_count += 1
    print(f"accuracy: {true_count / count}")


    if k != 0 and k % (epochs // 10) == 0:
        save_model(model, f"MNIST_Classifier_{k//10}")


# Get test data
with open("datasets/mnist_test.csv", mode="r", encoding="utf-8") as file:
    reader = csv.reader(file) 
    Xtest = []
    ytest = []
    header = True
    for row in reader:  
        if header:
            header = False
        else:
            temp_data = [] 
            for i in row:
                temp_data.append(int(i))
            Xtest.append([int(i) / 255.0 for i in row[1:]]) 
            ytest.append(temp_data[0])

# Calculate accuracy with test data
test_scores = [model(x) for x in Xtest]
test_score_idx = list(map(max_index, test_scores))

correct = sum(1 for i in range(len(test_score_idx)) if test_score_idx[i] == batch_y[i])
accuracy = correct / len(test_score_idx)
print(f"Test accuracy: {accuracy}") 

save_model(model, "puregrad/models/MNIST_Classifier")

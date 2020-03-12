from matplotlib import pyplot as plt 
import numpy as np 
import pickle
import os 
import torch


def loadPKL(path):
    with open(path, "rb") as f:
        return pickle.load(f)
        

def graph(arr, title, ylabel, show=False):
    plt.figure()
    plt.plot(list(range(len(arr))), arr)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.savefig("./graphs/" + title.replace(" ", "_") + ".png")
    if show:
        plt.show()




# MAML
train_accs_maml = loadPKL("./models/train_accs_maml.pkl")
train_losses_maml = loadPKL("./models/train_losses_maml.pkl")
train_losses_maml = [i.item() for i in train_losses_maml]
test_accs_maml = loadPKL("./models/test_accs_maml.pkl")
test_losses_maml = loadPKL("./models/test_losses_maml.pkl")
test_losses_maml = [i.item() for i in test_losses_maml]

print("MAML")
print(train_accs_maml)
print(train_losses_maml)
print(test_accs_maml)
print(test_losses_maml)

graph(train_accs_maml, "MAML Training Accuracy vs Iterations", "Accuracy", show=True)
graph(train_losses_maml, "MAML Training Loss vs Iterations", "Loss", show=True)

# pretrain
train_accs_pretrain = loadPKL("./models/train_accs_pretrain.pkl")
train_losses_pretrain = loadPKL("./models/train_losses_pretrain.pkl")
train_losses_pretrain = [i.item() for i in train_losses_pretrain]
test_accs_pretrain = loadPKL("./models/test_accs_pretrain.pkl")
test_losses_pretrain = loadPKL("./models/test_losses_pretrain.pkl")
test_losses_pretrain = [i.item() for i in test_losses_pretrain]


print("Pretrain")
print(train_accs_pretrain)
print(train_losses_pretrain)
print(test_accs_pretrain)
print(test_losses_pretrain)

graph(train_accs_pretrain, "Pretrain Training Accuracy vs Iterations", "Accuracy", show=True)
graph(train_losses_pretrain, "Pretrain Training Loss vs Iterations", "Loss", show=True)

# scratch
train_accs_ = loadPKL("./models/train_accs_.pkl")
train_losses_ = loadPKL("./models/train_losses_.pkl")
train_losses_ = [i.item() for i in train_losses_]
test_accs_ = loadPKL("./models/test_accs_.pkl")
test_losses_ = loadPKL("./models/test_losses_.pkl")
test_losses_ = [i.item() for i in test_losses_]

print("Scratch")
print(train_accs_)
print(train_losses_)
print(test_accs_)
print(test_losses_)

graph(train_accs_, "Scratch Training Accuracy vs Iterations", "Accuracy", show=True)
graph(train_losses_, "Scratch Training Loss vs Iterations", "Loss", show=True)

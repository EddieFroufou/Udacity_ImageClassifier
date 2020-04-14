#The project submission must include at least two files train.py and predict.py.
#The first file, train.py, will train a new network on a dataset and save the model as a checkpoint
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import seaborn as sns
#importing utility functions
from functions import inputs, data_loader, classifier, validate, test_model, save_model, train_network

#image mapping, getting it out of the way
import json
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

#Load inputs
inputs = inputs()
print(inputs)

#Assign inputs to variables // Not prompting user for a) saving directory and b) pre-trained model
data_dir = inputs.data_dir
learning_rate = inputs.learning_rate
hidden_layer1 = inputs.hidden_layer1
hidden_layer2 = inputs.hidden_layer2
dropout_prob1 = inputs.dropout_prob1
dropout_prob2 = inputs.dropout_prob2
epochs = inputs.epochs
use_gpu = inputs.gpu
print_every = inputs.print_every

# MAY THE GAMES BEGIN!
#1. LOAD PRE-TREATED DATA
train_loader, valid_loader, test_loader, train_data, valid_data, test_data = data_loader(data_dir)

#2. CLASSIFY DATA
model = classifier(hidden_layer1, hidden_layer2, dropout_prob1, dropout_prob2)

#3. TRAIN MODEL
model, optimizer, criterion = train_network(model, train_loader, valid_loader, epochs, print_every, learning_rate, use_gpu)

#4. TEST MODEL
test_model(model, test_loader, criterion, use_gpu)

#5. SAVE MODEL
save_model(model, epochs, optimizer, train_data)

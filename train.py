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
from functions import inputs, data_loader, classifier, validate, train_model, test_model, save_model

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

#defining pre-trained model
model = models.vgg16(pretrained=True)


# MAY THE GAMES BEGIN!
#1. LOAD PRE-TREATED DATA
train_loader, valid_loader, test_loader = data_loader(data_dir)

#2. CLASSIFY DATA
classifier(hidden_layer1, hidden_layer2, dropout_prob1, dropout_prob2)

#3. TRAIN MODEL
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

test_loss, accuracy = validate(model, train_loader, criterion, use_gpu)
model, optimizer = train_model(model, epochs, train_loader, valid_loader, criterion, optimizer, gpu_mode)

#4. TEST MODEL
test_model(model, test_loader, criterion, gpu_mode)

#5. SAVE MODEL
save_model(model, epochs, optimizer)

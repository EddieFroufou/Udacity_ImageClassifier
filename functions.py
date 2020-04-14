#---------------------------------------------------------------------------------------#
#TRAIN FUNCTIONS
#---------------------------------------------------------------------------------------#

#1. argparse
import argparse
#2. load and transform pictures/data
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
#3 and beyond
from torchvision import models
from torch import nn
from torch import optim

#image mapping, getting it out of the way
import json
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


#1. PARSER
def inputs():
    #assigning name to parser
    parser = argparse.ArgumentParser()

    #Prompting user for inputs, with default values already in place
    parser.add_argument('--data_dir', default='flowers', type=str, help="data directory")
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate (default = 0.001)')
    parser.add_argument('--hidden_layer1', default=6320, type=int, help='Inputs hidden layer 1 (default = 6320)')
    parser.add_argument('--hidden_layer2', default=1580, type=int, help='Inputs hidden layer 2 (default = 1580)')
    parser.add_argument('--dropout_prob1', default=0.5, type=float, help='Dropout probability 1 (default = 50%)')
    parser.add_argument('--dropout_prob2', default=0.5, type=float, help='Dropout probability 2 (default = 50%)')
    parser.add_argument('--epochs', default=3, type=int, help='Number of epochs for training (default = 3)')
    parser.add_argument('--gpu', default = True, action='store_true', dest='gpu', help='Use GPU for training')
    parser.add_argument('--print_every', default=5, type=int, help='Number of epochs for training (default = 5)')
    parser.add_argument('--image_path', default="./flowers/test/102/image_08015.jpg", type=str, help='Image path (default = "./flowers/test/102/image_08015.jpg"')
    parser.add_argument('--top_k', default=5, type=int, help='Number of outputs for prediction (default = 5)')
    
    #Consolidate inputs 
    inputs = parser.parse_args()

    return inputs

#2. DATA LOADER/TRANSFORMER
def data_loader(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    valid_transform = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    #Pass transforms in here
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=50, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=25, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=25, shuffle=True)
        
    return train_loader, valid_loader, test_loader, train_data, valid_data, test_data


#3. CLASSIFIER FUNCTION
def classifier(hidden_layer1, hidden_layer2, dropout_prob1, dropout_prob2):
    model = models.vgg16(pretrained = True)
    
    for param in model.parameters():
        param.requires_grad = False
        
    classifier = nn.Sequential(nn.Linear(25088, hidden_layer1, bias=True),
                               nn.ReLU(),
                               nn.Dropout(dropout_prob1),
                               nn.Linear(hidden_layer1, hidden_layer2, bias=True),
                               nn.ReLU(),
                               nn.Dropout(dropout_prob2),
                               nn.Linear(hidden_layer2, 102, bias=True),
                               nn.LogSoftmax(dim=1))
    
    model.classifier = classifier
    model.to('cuda')
    return model


#4. VALIDATION FUNCTION
def validate(model, data_loader, criterion):
    test_loss = 0
    accuracy = 0
    model.to('cuda')
        
    for ii, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy


#5. TRAIN CLASSIFIER FUNCTION
def train_network(model, dataloaders, validloaders, epochs, print_every,learning_rate=0.001, gpu=True):
    #define loss function
    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    steps=0
    #enable dropout in trainning dataset:
    model.train()
    # if GPU is available and GPU is set , then use GPU , or use CPU.
    if gpu == True:
        model.to('cuda')
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs,labels) in enumerate(dataloaders):
            steps += 1
            #use Variable to wrap the tensor and enable gradient descent.
            inputs.requires_grad_(True)
            labels.requires_grad_(True)
            #move tensor to target device(GPU|CPU)
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            #optimizer gradient to zero
            optimizer.zero_grad()
            #forward and backwards
            outputs=model.forward(inputs)
            loss=criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0 :
                model.eval()
                valid_loss,accuracy = validate(model,validloaders,criterion)
                print("Epoch {} / {}..".format(e+1,epochs),
                "loss: {:.4f}".format(running_loss/print_every),
                "Validation Loss: {:.3f}.. ".format(valid_loss),
                "Validation Accuracy: {:.3f}".format((accuracy)/len(validloaders)))
        running_loss=0
    return model, optimizer, criterion


#6. TEST MODEL FUNCTION
def test_model(model, test_loader, criterion, use_gpu):
    if use_gpu == True:
        model.to('cuda')
    else:
        pass
    
    model.eval()
    with torch.no_grad():
        test_loss, accuracy = validate(model, test_loader, criterion)
                
    print("Test Accuracy: {}%".format(accuracy*100/len(test_loader)))


#7. SAVE MODEL FUNCTION
def save_model(model, epochs, optimizer, train_data):
    model.cpu()
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'class_to_idx' : model.class_to_idx,
                  'classifier_state_dict': model.classifier.state_dict(),
                  'epochs': epochs,
                  'state_optimizer': optimizer.state_dict()}

    torch.save(checkpoint, 'checkpoint_part2.pth')
    
#---------------------------------------------------------------------------------------#
#PREDICT FUNCTIONS
#---------------------------------------------------------------------------------------#
import PIL
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
import torch.nn.functional as F


#8. LOAD MODEL FUNCTION
def load_checkpoint(filename, hidden_layer1, hidden_layer2, dropout_prob1, dropout_prob2):
    checkpoint = torch.load(filename)
    model = models.vgg16(pretrained=True)
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(nn.Linear(25088, hidden_layer1, bias=True),
                               nn.ReLU(),
                               nn.Dropout(dropout_prob1),
                               nn.Linear(hidden_layer1, hidden_layer2, bias=True),
                               nn.ReLU(),
                               nn.Dropout(dropout_prob2),
                               nn.Linear(hidden_layer2, 102, bias=True),
                               nn.LogSoftmax(dim=1))
    
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    model.classifier = classifier
    model.class_to_idx = checkpoint['class_to_idx']
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
    optimizer.load_state_dict(checkpoint['state_optimizer'])
    return (model, optimizer, criterion)


def process_image(image_path):
    img = PIL.Image.open(image_path)

    image_transform = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    tensor_image = image_transform(img)

    return tensor_image


def predict(image_tensor, model, cat_to_name, top_k):
    
    model.eval();

    torch_image = torch.from_numpy(np.expand_dims(image_tensor, axis=0)).type(torch.FloatTensor)

    model=model.cpu()

    log_probs = model.forward(torch_image)

    linear_probs = torch.exp(log_probs)

    top_probs, top_labels = linear_probs.topk(top_k)
    
    top_probs = np.array(top_probs.detach())[0]
    top_labels = np.array(top_labels.detach())[0]
    
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[flower] for flower in top_labels]
    top_flowers = [cat_to_name[flower] for flower in top_labels]
    
    return top_probs, top_labels, top_flowers

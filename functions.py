#1. argparse
import argparse
#2. load and transform pictures/data
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
#3 and beyond
from torchvision import models
from torch import nn



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
        
    return train_loader, valid_loader, test_loader


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
def train_model(model, epochs, train_loader, valid_loader, criterion, optimizer, use_gpu):
    steps = 0
    print_every = 10

    if use_gpu == True:
        model.to('cuda')
    else:
        pass
        
    for epoch in range(epochs):
        running_loss = 0
        
        for ii, (inputs, labels) in enumerate(train_loader):
            steps += 1        
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
            optimizer.zero_grad()        
       
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = validate(model, valid_loader, criterion)
            
                print(f"Epoch {epoch+1}/{epochs}..| "
                      f"Train loss: {running_loss/print_every:.3f}..| "
                      f"Validation loss: {valid_loss/print_every:.3f}..| "
                      f"Validation accuracy: {accuracy/len(valid_loader):.3f}|")
            
                running_loss = 0
                model.train()

    return model, optimizer


#6. TEST MODEL FUNCTION
def test_model(model, test_loader, criterion, use_gpu):
    if use_gpu == True:
        model.to('cuda')
    else:
        pass
    
    with torch.no_grad():
        test_loss, accuracy = validate(model, test_loader, criterion)
                
    print("Test Accuracy: {}%".format(accuracy*100/len(test_loader)))


#7. SAVE MODEL FUNCTION
def save_model(model, epochs, optimizer):
    model.cpu()

    checkpoint = {'class_to_idx' : model.class_to_idx,
                  'classifier_state_dict': model.classifier.state_dict(),
                  'epochs': epochs,
                  'state_optimizer': optimizer.state_dict()}

    torch.save(checkpoint, 'checkpoint_part2.pth')


#The second file, predict.py, uses a trained network to predict the class for an input image.
#Feel free to create as many other files as you need.
#Our suggestion is to create a file just for functions and classes relating to the model and 
#another one for utility functions like loading data and preprocessing images.
from functions import load_checkpoint, process_image, predict, inputs
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np
import torch.nn.functional as F
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
image_path = inputs.image_path
top_k = inputs.top_k
use_gpu = inputs.gpu

#image processing and predicting

model , optimizer, criterion = load_checkpoint('checkpoint_part2.pth', hidden_layer1, hidden_layer2, dropout_prob1, dropout_prob2)

tensor_image = process_image(image_path)

top_probs, top_labels, top_flowers = predict(tensor_image, model, cat_to_name, top_k)

for i in range(0, top_k):
    print("Likelihood #{}, with probability {:.4%}, is that this flower is a {} that can be seen in folder with label {}".format(i+1, top_probs[i], top_flowers[i], top_labels[i]))

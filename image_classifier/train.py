import argparse

from torch import optim
import torch.nn as nn
from torchvision import models

from utils import load_transform_data

from functions import generate_classifier, train_model, test_model, save_model


parser = argparse.ArgumentParser(
    description ='Image Classification - Training Neural Network'
)

parser.add_argument('data_directory', action='store',
                    help='Set path to data folder.')

parser.add_argument('-arch', '--architecture', action='store',
                    dest='pretrained_model', default='vgg11',
                    help='Select pretrained neural network model.')

parser.add_argument('-s', '--save_dir', action='store',
                    dest='save_directory', default='checkpoint.pth',
                    help='Set directory to save checkpoint.')

parser.add_argument('-hu', '--hidden_units', action='store',
                    dest='hidden_units', type=int, default=500,
                    help='Set number of hidden layer units')

parser.add_argument('-do', '--dropout', action='store',
                    dest='dropout', type=float, default=0.1,
                    help='Set dropout value to be used in training the NN')

parser.add_argument('-lr', '--learning_rate', action='store',
                    dest='learning_rate', type=float, default = 0.001,
                    help='Set learning rate for optimizer')

parser.add_argument('-e', '--epochs', action='store',
                    dest='epochs', type=int, default=4,
                    help='Set number of epochs being the number of passes through the entire dataset')

parser.add_argument('--gpu', action='store_true', default=False,
                    help='GPU flag. If --gpu is specified, this assigns the value to True')

results = parser.parse_args()

# Assign parser arguments to new variables
data_dir = results.data_directory
save_dir = results.save_directory
pretrained_model = results.pretrained_model
hidden_units = results.hidden_units
dropout = results.dropout
learn_rate = results.learning_rate
epochs = results.epochs
gpu_switch = results.gpu

# Load data from ImageFolder and apply transformations
trainloader, testloader, validationloader, train_data, test_data, validation_data = load_transform_data(data_dir)

# Load a pre-trained network
model = getattr(models, pretrained_model)(pretrained=True)

# Define a new, untrained feed-forward network as a classifier of the pretrained model
input_units = model.classifier[0].in_features
generate_classifier(model, input_units, hidden_units, dropout)

# Define criterion for loss calculation
# Since output from classifier - LogSoftmax --> loss calculation - nn.NLLLoss
criterion = nn.NLLLoss()

# Define Optimizer
optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)

# Train the NN classifier model
model, optimizer = train_model(model, epochs, trainloader, validationloader, optimizer, criterion, gpu_switch)

# Validation on the test set
test_model(model, testloader, criterion, gpu_switch)

# Save the model
save_model(model, train_data, optimizer, epochs, save_dir)

# type on command line:
# python train.py ./flowers -arch vgg11 --gpu

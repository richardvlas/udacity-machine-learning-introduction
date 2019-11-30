import argparse
import json

from torchvision import models

from functions import load_pretrained_model, predict

from utils import process_image


parser = argparse.ArgumentParser(
    description ='Image Classification - Predict using Trained Neural Network'
)

parser.add_argument('path_to_image', action='store',
                    help='Set path to an image.')

parser.add_argument('checkpoint', action='store',
                    help='Set path to checkpoint.')

parser.add_argument('-arch', '--architecture', action='store',
                    dest='pretrained_model', default='vgg11',
                    help='Select pretrained neural network model.')

parser.add_argument('--top_k', action='store',
                    dest='top_k', type=int, default=4,
                    help='Set number of predicted classes with highest probabilities.')

parser.add_argument('-cn', '--category_names', action='store',
                    dest='cat_names', default='cat_to_name.json',
                    help='Set path to file for mapping from category label to category name.')

parser.add_argument('--gpu', action='store_true', default=False,
                    help='GPU flag. If --gpu is specified, this assigns the value to True')

results = parser.parse_args()

# Assign parser arguments to new variables
image = results.path_to_image
checkpoint = results.checkpoint
pretrained_model = results.pretrained_model
top_k = results.top_k
cat_names = results.cat_names
gpu_switch = results.gpu

# Load in a mapping from category label to category name
with open(cat_names, 'r') as f:
    cat_to_name = json.load(f)

# Load a pre-trained network
model = getattr(models, pretrained_model)(pretrained=True)

# Load model checkpoint
pretrained_model = load_pretrained_model(model, checkpoint)

# Image Preprocessing
processed_img = process_image(image)

# Class Prediction
top_probs, top_cats, top_names = predict(processed_img, model, top_k, cat_to_name, gpu_switch)

# Print flower name along with the probability of that name
print(f'\n\nTop {top_k} probabilities:')
print(f'---------------------------------')

for prob, name in zip(top_probs, top_names):
    print(f'Probability of {name} is {prob*100:6.1f}%')

# type on command line:
# python predict.py flowers/test/28/image_05230.jpg checkpoint.pth


import numpy as np

import torch
import torch.nn as nn

from collections import OrderedDict
import time



def generate_classifier(model, input_units, hidden_units, dropout):
    ''' Define a new, untrained feed-forward network as a classifier
    '''

    # Model parameters of the pretrained network are frozen
    # therefore not updated in backpropagation steps
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                           ('fc1', nn.Linear(input_units, hidden_units)),
                           ('relu', nn.ReLU()),
                           ('drop1', nn.Dropout(dropout)),
                           ('fc2', nn.Linear(hidden_units, 102)),
                           ('output', nn.LogSoftmax(dim=1))
                           ]))

    # Save the new classifier in the model (replace the default classifier)
    model.classifier = classifier



def train_model(model, epochs, trainloader, validationloader, optimizer, criterion, gpu_switch):
    ''' Train the classifier layers
    '''

    if (gpu_switch == True) and (torch.cuda.is_available() == True):
        device = torch.device('cuda')
        model.to(device)

    elif (gpu_switch == True) and (torch.cuda.is_available() == False):
        print('GPU not available. Set to \'cpu\'')
        device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    # Training information:

    # Training Device
    print(f"Training Device: {device}")

    # Total Number of images in training dataset
    train_nr_img = len(trainloader.dataset)
    print(f"Number of images in training dataset: {train_nr_img}")

    # Number of images in trainloader per batch
    nr_img_batch = trainloader.batch_size
    print(f"Number of images per batch: {nr_img_batch}")

    # Number of batches in dataset:
    nr_batch = len(trainloader)
    print(f"Number of batches (of size {nr_img_batch}): {nr_batch}")

    #############################
    # Train the classifier layers
    #############################
    print(f"\nTraining the classifier layers:")

    #epochs = 4
    steps = 0
    print_every = 5

    start_time = time.time()
    # epoch = one pass through the entire dataset
    for epoch in range(epochs):
        start_epoch_time = time.time()
        running_loss = 0
        nr_labels = 0
        equals = 0
        # assign first batch number
        bns = 1

        print(f"\n----------------\nEpoch {epoch+1}/{epochs}\n----------------")

        for inputs, labels in trainloader:
            steps += 1

            # Move input and label tensors to the specified device
            inputs, labels = inputs.to(device), labels.to(device)

            # Clear the gradients, so they dont' get accumulated
            optimizer.zero_grad()

            # TODO: Training pass
            # Forward pass -> backward pass -> weight's update
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            # Training loss
            running_loss += loss.item()

            # Training Accuracy
            _, pred = torch.max(logps.data, 1)
            nr_labels += len(labels)
            equals += torch.sum(pred == labels).item()


            if steps % print_every == 0:

                average_loss = f'Train loss: {running_loss/steps:.2f}'
                test_acc = f'Accuracy: {(equals/nr_labels * 100):.1f}%'
                print(f'   Batches {bns:03} - {steps:03} | {average_loss} | {test_acc}')

                bns = steps + 1

        steps = 0


        test_loss = 0
        accuracy = 0
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            model.eval()
            for inputs, labels in validationloader:

                # Move input and label tensors to the specified device
                # this won't copy if they are already on the desired device
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)

                test_loss += batch_loss #.item()

                # Calculate validation accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()


        val_acc = f'{(accuracy/len(validationloader) * 100):.1f}% of images accurately predicted'

        epoch_time = time.time() - start_epoch_time

        print(f"\n\tValidation Results: {val_acc}")
        print(f'\n\tEpoch {epoch+1} time: {epoch_time:6.1f} s')

        # Putting model back into training mode
        model.train()


    total_time = time.time() - start_time
    tot_time = time.strftime('%H:%M:%S', time.gmtime(total_time))

    print(f'\n\n*** Calculation Done! ***')
    print(f'---------------------------------')
    print(f'Total Elapsed Time: {tot_time}')

    return model, optimizer



def test_model(model, testloader, criterion, gpu_switch):
    ''' Do validation on the test set
    '''

    if (gpu_switch == True) and (torch.cuda.is_available() == True):
        device = torch.device('cuda')
        model.to(device)

    elif (gpu_switch == True) and (torch.cuda.is_available() == False):
        print('GPU not available. Set to \'cpu\'')
        device = torch.device('cpu')
    else:
        device = torch.device('cpu')


    test_loss = 0
    accuracy = 0
    nr_labels = 0

    with torch.no_grad():
        model.eval()
        for inputs,labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model(inputs)
            test_loss += criterion(logps, labels)
            nr_labels += len(labels)

            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

    print(f'Test Accuracy: {accuracy/len(testloader)*100:.1f}% of {nr_labels} images.')



def save_model(model, train_data, optimizer, epochs, save_dir):
    ''' Save the checkpoint
    '''

    # Create dictionary with all information
    checkpoint = {'state_dict': model.state_dict(),
                  'classifier': model.classifier,
                  'class_to_idx': train_data.class_to_idx,
                  'optimizer_state': optimizer.state_dict,
                  'n_epoch': epochs
                  }

    torch.save(checkpoint, save_dir)

    print('\nModel Saved!')



def load_pretrained_model(model, filepath):
    ''' Load a model checkpoint
    '''
    checkpoint = torch.load(filepath)

    for param in model.parameters():
        param.requires_grad=False

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model



def predict(processed_image, model, topk, cat_to_name, gpu_switch):
    ''' Predict the class of an image using a trained deep learning model.
    '''

    image = processed_image.unsqueeze(dim=0)

    if gpu_switch == True:
        model.to('cuda')
        image = image.to('cuda')
    else:
        model.to('cpu')

    model.eval()

    with torch.no_grad():

        logps = model.forward(image)

        ps = torch.exp(logps)

        top_probs, top_idx = torch.topk(ps, topk)

        top_probs, top_idx = top_probs.cpu(), top_idx.cpu()

        top_probs = top_probs.numpy()[0,:]
        top_idx = top_idx.numpy()[0,:]

        index_to_class = {val: key for key, val in model.class_to_idx.items()}
        top_cats = [index_to_class[top_id] for top_id in top_idx]
        top_names = [cat_to_name[cat] for cat in top_cats]

        return top_probs, top_cats, top_names

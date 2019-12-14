import argparse
import torch
import numpy as np
import sys
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms, models, datasets
from collections import OrderedDict
from PIL import Image
from typing import Any


def main():
    print("running train.py................")
    args = get_args()

    data_dir = args.data_dir
    save_dir = args.save_dir
    arch = args.arch
    learning_rate = args.learning_rate
    ep = args.epochs
    hidden_units = args.hidden_units
    device = args.gpu
    global gpu
    gpu = device
    if gpu:
        print("Using gpu")
    else:
        print("No gpu")

    # Define Model
    if arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_size = 1024
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)
        input_size = 25088
    else:
        print('{} is unsupported by current application'.format(arch))
        sys.exit()

    # load and process images from data directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    trainloader, validloader, testloader, train_data = load_data(train_dir, valid_dir, test_dir)

    # Freeze parameters
    for parameter in model.parameters():
        parameter.requires_grad = False

    # define criterion and optimizer
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout', nn.Dropout(0.3)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    print("Training Loss:...")
    train(model, trainloader, validloader, criterion, optimizer, ep, gpu)

    print("Testing Accuracy(using test data):....")
    test_accuracy(testloader, model, criterion, gpu)

    # save to checkpoint

    model.class_to_idx = train_data.class_to_idx
    torch.save({'structure': arch,
                'epochs': ep,
                'input_size': input_size,
                'hidden_layer1': hidden_units,
                'optimizer': optimizer,
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx, }, save_dir)


def get_args():
    ##Get arguments from command line
    parser = argparse.ArgumentParser()  # type: Any

    parser.add_argument('data_dir', type=str, help="data directory that has train, valid and test sub-folders")
    parser.add_argument('--arch', type=str, default="vgg19", help="pre-trained model: vgg19,densenet121")
    parser.add_argument('--gpu', action='store_true', default=False, help="Use GPU")
    parser.add_argument('--epochs', type=int, default=1, help="number of epochs to train model")
    parser.add_argument('--hidden_units', type=int, default=1024, help="list of hidden layers")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--save_dir', type=str, default="checkpoint_com.pth",
                        help="directory for the Checkpoint file to save")

    return parser.parse_args()


def load_data(train_dir, valid_dir, test_dir):
    # Load Data

    # Transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=train_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

    return trainloader, validloader, testloader, train_data


def train(model, trainloader, validloader, criterion, optimizer, epochs, gpu):
    epochs = epochs
    steps = 0
    running_loss = 0
    print_every = 20
    valid_len = len(validloader)
    # change to cuda
    if gpu:
        model.to('cuda')

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # move input and label tensors to the GPU
            if gpu:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()

            # Forward and backward passes
            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion)

                print(f"Epoch {epoch + 1}/{epochs}.. ",
                      f"Loss: {running_loss / print_every:.3f}.. ",
                      f"valid_loss: {valid_loss / valid_len:.3f}.. ",
                      f"Accuracy: {accuracy / valid_len:.3f}")

                running_loss = 0
                model.train()


def validation(model, validloader, criterion):
    valid_loss = 0
    accuracy = 0
    for inputs, labels in validloader:
        if torch.cuda.is_available() and gpu:
            ##print(" Using GPU.........")
            model.to('cuda')
            inputs, labels = inputs.to('cuda'), labels.to('cuda')

        logps = model.forward(inputs)
        valid_loss += criterion(logps, labels).item()

        # Calculate accuracy
        ps = torch.exp(logps)
        top_ps, top_class = ps.topk(1, dim=1)
        equals = (top_class == labels.view(*top_class.shape))
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    return valid_loss, accuracy


def test_accuracy(testloader, model, criterion, gpu):
    # Testing network

    test_loss = 0
    accuracy = 0
    if torch.cuda.is_available() and gpu:
        model.cuda()

    model.eval()

    with torch.no_grad():
        for inputs, labels in testloader:
            if torch.cuda.is_available() and gpu:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test accuracy: {accuracy / len(testloader):.3f}")


# Run the program

if __name__ == "__main__":
    main()
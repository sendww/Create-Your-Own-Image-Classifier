import argparse
import torch
import numpy as np
import sys
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms, models, datasets
from collections import OrderedDict
from PIL import Image
from typing import Any
import json


def main():
    # get arguments from command line
    arg = get_args()

    path_to_image = arg.image_path
    checkpoint_ = arg.checkpoint
    num = arg.top_k
    cat_names = arg.category_names
    device = arg.gpu
    global gpu
    gpu = device
    # load category names file
    with open(cat_names, 'r') as f:
        cat_to_name = json.load(f)

    # load trained model
    model = load(checkpoint_)
    '''
    class_code = path_to_image.split('/')[3]
    flower_name = cat_to_name[class_code]
    print("flow name of test image(class code:{})  is {}........".format(class_code, flower_name))
    '''

    # Process images, predict classes, and display results
    # image = process_image(path_to_image)
    probs, classes = predict(path_to_image, model, num)
    prob = np.array(probs)
    ##prob = np.array(probs.cpu().numpy())
    labels = [cat_to_name[index] for index in classes]
    top = 0
    while top < num:
        print("{} with a probability of {}".format(labels[top], prob[top]))
        top += 1
    print("Finish prediction ")


def load(checkpoint_):
    # load checkpoint file

    checkpoint = torch.load(checkpoint_)

    # model = getattr(torchvision.models, checkpoint['structure'])(pretrained=True)
    if checkpoint['structure'] == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif checkpoint['structure'] == 'vgg19':
        model = models.vgg19(pretrained=True)

    else:
        print('{} is unsupported by current application'.format(arch))
        sys.exit()

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(checkpoint['input_size'], checkpoint['hidden_layer1'])),
        ('relu1', nn.ReLU()),
        ('dropout', nn.Dropout(0.3)),
        ('fc2', nn.Linear(checkpoint['hidden_layer1'], 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
    if torch.cuda.is_available() and gpu:
        ##print(" Using GPU.........")
        model.to('cuda')

    model.class_to_idx = checkpoint['class_to_idx']

    model.load_state_dict(checkpoint['state_dict'])
    model.classifier.optimizer = checkpoint['optimizer']
    # model.classifier.epochs = checkpoint['epochs']
    # model.classifier.learning_rate = checkpoint['learning_rate']
    return model


def get_args():
    ##Get arguments from command line
    parser = argparse.ArgumentParser()

    parser.add_argument("image_path", type=str, help="path to image in which to predict class label")
    parser.add_argument("checkpoint", type=str, help="checkpoint in which trained model is contained")
    ## temp1
    parser.add_argument("--top_k", type=int, default=5, help="number of classes to predict")
    parser.add_argument("--category_names", type=str, default="cat_to_name.json",
                        help="file to convert label index to label names")
    # parser.add_argument('--gpu', action='store_true', default=False, help="Use GPU")
    parser.add_argument('--gpu', action='store_true', default=False, help="Use GPU")

    return parser.parse_args()


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    pil_img = Image.open(image)
    process_img = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    nd_image = process_img(pil_img)

    return nd_image


def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if torch.cuda.is_available() and gpu:
        print(" Using GPU.........")
        model.to('cuda')
    else:
        print("Without using GPU.....")

    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()

    if gpu:
        with torch.no_grad():
            output = model.forward(img_torch.cuda())
    else:
        with torch.no_grad():
            output = model.forward(img_torch)

    probability = F.softmax(output.data, dim=1)
    probs, indices = probability.topk(topk)
    probs = probs.cpu().numpy()[0]
    indices = indices.cpu().numpy()[0]

    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[x] for x in indices]
    print(probs)
    print(classes)
    return probs, classes


# Run the program
if __name__ == "__main__":
    main()

# Imports here
'''%matplotlib inline'''
'''%config InlineBackend.figure_format = 'retina' '''

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json

import numpy as np
import PIL
from PIL import Image

# Use GPU if it's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def transform_images(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])


    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir,transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir,transform=test_transforms)
    valid_data = datasets.ImageFolder(valid_dir,transform=valid_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data,batch_size=64,shuffle=True) 
    testloader = torch.utils.data.DataLoader(test_data,batch_size=64,shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data,batch_size=64,shuffle=True)
    
    return train_data,test_data, valid_data, trainloader, testloader, validloader


def network_model(structure='vgg13',dropout=0.5, hidden_layer1 = 4096,learning_rate = 0.0001):
    # TODO: Build and train your network
    if structure =='vgg13':
        model = models.vgg13(pretrained=True)
    elif structure =='vgg16':
        model = models.vgg16(pretrained=True)
    else:
        print("Please try for vgg13 or vgg16 only")

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(nn.Linear(25088, hidden_layer1),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(hidden_layer1, 102),
                                     nn.LogSoftmax(dim=1))
    criterion = nn.NLLLoss()
    
    optimizer = optim.Adam(model.classifier.parameters(),lr=learning_rate)
    model.to(device)
    
    return model, criterion, optimizer


def deep_learning(model, criterion, optimizer, epochs=2, print_every = 5, trainloader=0, validloader=0):
    
    steps = 0
    running_loss = 0

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps +=1

            inputs, labels = inputs.to(device), labels.to(device)

            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(validloader):.3f}.. "
                      f"Test accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()


def save_checkpoint(train_data=0,model=0,path='checkpoint.pth',structure ='vgg13', hidden_layer1 = 4096,dropout=0.5,learning_rate=0.0001,epochs=2):

    checkpoint = {'structure' :structure,
                  'hidden_layer1':hidden_layer1,
                  'dropout':dropout,
                  'epochs':epochs,
                  'state_dict': model.state_dict(),
                  'classifier': model.classifier,
                  'learning_rate': learning_rate,
                  'class_to_idx': train_data.class_to_idx,
                  'optimizer_dict': optimizer.state_dict()}
    torch.save(checkpoint, path)


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    structure = checkpoint['structure']
    dropout = checkpoint['dropout']
    hidden_layer1 = checkpoint['hidden_layer1']
    learning_rate = checkpoint['learning_rate']
    model, criterion, optimizer = network_model(structure,dropout,hidden_layer1,learning_rate)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return optimizer,criterion, model


def process_image(image):
    pil_image = PIL.Image.open(image)
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    
    np_image = transform(pil_image)
        
    return np_image


def predict(image_path, model, topk=5):
    load_checkpoint('checkpoint.pth')
    
    model.idx_to_class = dict(map(reversed, model.class_to_idx.items()))
    
    with torch.no_grad():
        image = process_image(image_path).unsqueeze(dim=0)
        image = image.float()
        outputs = model.forward(image.cuda())
        ps = torch.exp(outputs)
        probs, indices = ps.topk(topk)
        #probs = probs.squeeze()
        classes = [model.idx_to_class[idx] for idx in indices[0].tolist()]
        
    return probs, classes



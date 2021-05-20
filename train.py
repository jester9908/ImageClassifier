import argparse
import os

import image_classifier as iutils

ap = argparse.ArgumentParser(description='Train.py')


ap.add_argument('data_dir', action="store", default="./flowers/")
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./")
ap.add_argument('--arch', dest="arch", action="store", default="vgg13", type = str)
ap.add_argument('--learning_rate',type=float, dest="learning_rate", action="store", default=0.0001)
ap.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=4096)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=2)
ap.add_argument('--gpu', dest="gpu", action="store", default="gpu")
ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)

pa = ap.parse_args()
root = pa.data_dir
path = pa.save_dir
structure = pa.arch
learning_rate = pa.learning_rate
hidden_layer1 = pa.hidden_units
epochs = pa.epochs
dev = pa.gpu
dropout = pa.dropout


def main():
    if not os.path.exists(root):
        print("The data directory is not available")
    elif not (structure=="vgg13" or structure=="vgg16"):
        print("Please try for vgg13 or vgg16 only")
    else:
        
        new_path = iutils.create_dir(path)
        device = iutils.check_device(dev)
        train_data, test_data, valid_data, trainloader, testloader, validloader = iutils.transform_images(root)
        model, criterion, optimizer = iutils.network_model(structure=structure, hidden_layer1=hidden_layer1,learning_rate=learning_rate, device=device)
        iutils.deep_learning(model, criterion, optimizer, epochs=epochs, trainloader=trainloader, validloader=validloader,device=device)
        iutils.save_checkpoint(train_data, model,new_path,structure, hidden_layer1,dropout,learning_rate,epochs)
        print("Training Completed Succesfully!")


if __name__== "__main__":
    main()
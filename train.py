import argparse

import image_classifier as iutils

ap = argparse.ArgumentParser(description='Train.py')


ap.add_argument('data_dir', action="store", default="./flowers/")
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
ap.add_argument('--arch', dest="arch", action="store", default="vgg13", type = str)
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
ap.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=4096)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
ap.add_argument('--gpu', dest="gpu", action="store", default="gpu")
ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)

pa = ap.parse_args()
root = pa.data_dir
path = pa.save_dir
structure = pa.arch
learning_rate = pa.learning_rate
hidden_layer1 = pa.hidden_units
epochs = pa.epochs
device = pa.gpu
dropout = pa.dropout


def main():    
    train_data, test_data, valid_data, trainloader, testloader, validloader = iutils.transform_images(root)
    model, criterion, optimizer = iutils.network_model(structure=structure, hidden_layer1=hidden_layer1,learning_rate=learning_rate)
    iutils.deep_learning(model, criterion, optimizer, epochs=epochs, trainloader=trainloader, validloader=validloader)
    iutils.save_checkpoint(model,path,structure, hidden_layer1,dropout,learning_rate,epochs, train_data)
    print("Training Completed Succesfully!")


if __name__== "__main__":
    main()
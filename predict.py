import argparse
import json
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import image_classifier as iutils

ap = argparse.ArgumentParser(description='Predict.py')
ap.add_argument('image', action="store", default='./flowers/test/1/image_06743.jpg')
ap.add_argument('checkpoint', action="store", type = str, default='./checkpoint.pth')
ap.add_argument('--top_k', dest="top_k", action="store", type=int, default=5 )
ap.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
ap.add_argument('--gpu', default="gpu", action="store", dest="gpu")

pa = ap.parse_args()
inputpath = pa.image
filepath = pa.checkpoint
number_of_outputs = pa.top_k
dev = pa.gpu

def main():
    newinputpath=iutils.select_file_name(inputpath)
    optimizer,criterion, model = iutils.load_checkpoint(filepath)
    with open('cat_to_name.json', 'r') as json_file:
        cat_to_name = json.load(json_file)
    probs, classes =  iutils.predict(newinputpath, model, number_of_outputs)
    flower_classes = [cat_to_name[str(cls)] + "({})".format(str(cls)) for cls in classes]
    probability = np.array(probs[0])
    i=0
    while i < number_of_outputs:
        print("{} with a probability of {}".format(flower_classes[i], "{:.0%}". format(probability[i])))
        i += 1
    print("Prediction Completed!")
    
    

if __name__== "__main__":
    main()
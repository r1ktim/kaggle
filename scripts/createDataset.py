import sys
import numpy as np
import cv2
import os

path = "/home/rik/KaggleData/dogsVsCats/train";

trainData = [];
#trainData = np.array(trainData);
resizeSize = 52;

for dirpath, dirnames, filenames in os.walk(path):
    for f in filenames:
        img = cv2.imread(os.path.join(path, f));
        img = cv2.resize(img, (resizeSize, resizeSize));
        
        img = img[:, :, :] / 255.
                
        if "dog" in f:
            label = [1, 0];
        elif "cat" in f:
            label = [0, 1];
        else:
            print "Not a correct label?";
            exit(0);
            
        imgFlip1 = cv2.flip(img.copy(), 0);
        imgFlip2 = cv2.flip(img.copy(), 1);
        imgFlip3 = cv2.flip(img.copy(), -1);
        
        trainData.append([img, label]);  
        trainData.append([imgFlip1, label]);
        trainData.append([imgFlip2, label]);
        trainData.append([imgFlip3, label]);
       
trainData = np.asarray(trainData); 
print len(trainData);
np.save("/home/rik/KaggleData/dogsVsCats/training.npy", trainData);


path = "/home/rik/KaggleData/dogsVsCats/test";

trainData = [];
#trainData = np.array(trainData);

for dirpath, dirnames, filenames in os.walk(path):
    for f in filenames:
        img = cv2.imread(os.path.join(path, f));
        img = cv2.resize(img, (resizeSize, resizeSize));
        
        img = img[:, :, :] / 255.
                
        f = f.strip(".jpg");
        
        trainData.append([img, f]);  
       
trainData = np.asarray(trainData); 
print len(trainData);
np.save("/home/rik/KaggleData/dogsVsCats/test.npy", trainData);
print 'Done';  
        

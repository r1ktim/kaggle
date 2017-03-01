import numpy as np
from network import Network
import sys
import time
import math
import random

network = Network();
network.CreateNetwork();

maxEpochs = 10000;
batch_size = 256;

lr = 0.1;
minLr = 0.00001;
decay = 0.99;
begin = time.time();

data = np.load("/home/rik/KaggleData/dogsVsCats/training.npy");
print len(data);

trainInput = [];
trainLabels = [];

indices = range(len(data));
random.shuffle(indices);

for i in indices:
    trainInput.append(data[i][0]);
    trainLabels.append(data[i][1]);

def logloss(pred, act):
    pred = pred[0];
    act = act[0]
    epsilon = 1e-15
    pred = max(epsilon, pred)
    pred = min(1-epsilon, pred)
    
    if 1 - pred == 0:
	pred += 1e-15;
    ll = act*math.log(pred) + (1 -act)*math.log(1-pred)
    
    return ll

def CalcLogLoss(output, label):
    loss = 0;
    try:
        loss = label[0] * math.log(output[0] +0.0000000001) + (1 - label[0])*math.log(1 - output[0] +0.00000001)
    except Exception as e:
        print 'Error in CalcLogLoss';
        print e;
        print output[0];
    
    return loss
   
currentBatchIndex = 0; 
for e in range(maxEpochs):
    
    for i in range(0, len(trainInput) / 10, batch_size):
        newBatchIndex = currentBatchIndex + batch_size;       
    
        if newBatchIndex >= len(trainInput):
            newBatchIndex -= newBatchIndex - len(trainInput);
        
        trainBatch = trainInput[currentBatchIndex: newBatchIndex];
        targetBatch = trainLabels[currentBatchIndex : newBatchIndex];
        
        if newBatchIndex == len(trainInput):
            currentBatchIndex = 0;
        else:
            currentBatchIndex = newBatchIndex;
            
        network.Train(trainBatch, targetBatch, lr);
        
    if (e % 20 == 0): # Calculate the training error
        print 'Epoch:', e
        loss = 0;
        count = 0;
        for i in range(0, len(trainInput)/ 10):   
            
            output =  network.Run([trainInput[i]])[0];
            loss += logloss(output, trainLabels[i]);
           
        print 'Training accuracy:', -(loss / (len(trainInput)/10));      
        network.Save("/home/rik/models/dogsVsCats/", str(e)); 
        
    if lr > minLr:
        lr *= decay;
        if (lr < minLr):
            lr = minLr;

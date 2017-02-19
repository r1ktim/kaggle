import pandas as pd
import numpy as np
from network import Network

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def CreateLabels(x):
    labels = [];
    for l in x:
        label = np.zeros(10);
        label[l] = 1;
        labels.append(label);
    
    return labels;
       

df = pd.read_csv("/home/rik/KaggleData/mnist/train.csv");

data = df.as_matrix();


y = data[:, 0];
x = data[:, 1:].astype(np.float32) / 255;

totalNrOfSamples =  x.shape[0];
totalNrOfVal = 1000;
totalNrOfTrain = totalNrOfSamples - totalNrOfVal;
print 'Total Training examples:', totalNrOfTrain
print 'Total Validiation examples:', totalNrOfVal

trainInput = x[:totalNrOfTrain];
trainTarget = CreateLabels(y[:totalNrOfTrain]);
valInput = x[totalNrOfTrain:];
valTarget = CreateLabels(y[totalNrOfTrain:]);
testInput = mnist.test.images;
testTarget = mnist.test.labels;

network = Network();
network.CreateNetwork();

maxEpochs = 10000;
batch_size = 256;

for e in range(maxEpochs):
    
    if e % 10 == 0:
        trainEval = [];
        for t in range(0, len(trainInput), batch_size):
            if (t+batch_size < len(trainInput)):
                batchInput = trainInput[t:t+batch_size];
                batchTarget = trainTarget[t:t+batch_size];    
            else:
                batchInput = trainInput[t:t + abs((abs(len(trainInput) -(t + batch_size)) - batch_size))]
                batchTarget = trainTarget[t:t + abs((abs(len(trainTarget) -(t + batch_size)) - batch_size))];
            trainEval.append(network.Eval(batchInput, batchTarget));
        
        valEval = [];
        
        for t in range(0, len(valInput), batch_size):
            if (t+batch_size < len(valInput)):
                batchInput = valInput[t:t+batch_size];
                batchTarget = valTarget[t:t+batch_size];    
            else:
                batchInput = valInput[t:t + abs((abs(len(valInput) -(t + batch_size)) - batch_size))]
                batchTarget = valTarget[t:t + abs((abs(len(valTarget) -(t + batch_size)) - batch_size))];
                
            valEval.append(network.Eval(batchInput, batchTarget));
        if (e % 100 == 0): 
            testEval = [];
            
            for t in range(0, len(testInput), batch_size):
                if (t+batch_size < len(testInput)):
                    batchInput = testInput[t:t+batch_size];
                    batchTarget = testTarget[t:t+batch_size];    
                else:
                    batchInput = testInput[t:t + abs((abs(len(testInput) -(t + batch_size)) - batch_size))]
                    batchTarget = testTarget[t:t + abs((abs(len(testTarget) -(t + batch_size)) - batch_size))];
                    
                testEval.append(network.Eval(batchInput, batchTarget));
        
        
        print "Epoch:", e;
        print "Train correct:", np.mean(trainEval);
        print "Val correct:", np.mean(valEval);
        if (e % 100 == 0):
            print "Test correct:", np.mean(testEval);
        
    for t in range(0, len(trainInput), batch_size):
        if (t+batch_size < len(trainInput)):
            batchInput = trainInput[t:t+batch_size];
            batchTarget = trainTarget[t:t+batch_size];

        else:
            batchInput = trainInput[t:t + abs((abs(len(trainInput) -(t + batch_size)) - batch_size))]
            batchTarget = trainTarget[t:t + abs((abs(len(trainTarget) -(t + batch_size)) - batch_size))];

        network.Train(batchInput, batchTarget);

import pandas as pd
import numpy as np
from convNetwork import Network
import sys

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def CreateLabels(x):
    labels = [];
    for l in x:
        label = np.zeros(10);
        label[l] = 1;
        labels.append(label);
    
    return labels;

network = Network();
network.CreateNetwork();

if (len(sys.argv) > 1):
    network.LoadConv("/home/rik/models/", sys.argv[1] + ".ckpt");
else:
    print "No model provided";
    exit(0);
    
df = pd.read_csv("/home/rik/KaggleData/mnist/train.csv");

data = df.as_matrix();


y = data[:, 0];
x = data[:, 1:].astype(np.float32) / 255;

testInput = mnist.test.images;


totalNrOfSamples =  x.shape[0];
totalNrOfVal = 0;
totalNrOfTrain = totalNrOfSamples - totalNrOfVal;
print 'Total Training examples:', totalNrOfTrain
print 'Total Validiation examples:', totalNrOfVal

trainInput = x[:totalNrOfTrain];
trainTarget = CreateLabels(y[:totalNrOfTrain]);

featureOutput = []

for i in range(totalNrOfTrain):
    output = network.Run([x[i]]);
    featureOutput.append(output[0]);
    
features = np.asarray(featureOutput);

print (features[0] == 0).sum();

np.save("/home/rik/KaggleData/mnist/featuresTrain.npy", features);

featureOutput = []

for i in range(len(testInput)):
    output = network.Run([testInput[i]]);
    featureOutput.append(output[0]);
    
features = np.asarray(featureOutput);

print (features[0] == 0).sum();

np.save("/home/rik/KaggleData/mnist/featuresTest.npy", features);




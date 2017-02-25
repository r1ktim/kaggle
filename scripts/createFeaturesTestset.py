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
    print 'No model provided';
    exit(0);
    
df = pd.read_csv("/home/rik/KaggleData/mnist/test.csv");

data = df.as_matrix();


x = data[:, 0:].astype(np.float32) / 255;



totalNrOfSamples =  x.shape[0];
totalNrOfVal = 0;
totalNrOfTrain = totalNrOfSamples - totalNrOfVal;
print 'Total test examples:', totalNrOfTrain


trainInput = x[:totalNrOfTrain];

featureOutput = []

for i in range(len(x)):
    output = network.Run([x[i]]);
    featureOutput.append(output[0]);
    
features = np.asarray(featureOutput);

print (features[0] == 0).sum();

np.save("/home/rik/KaggleData/mnist/featuresFinalTest.npy", features);






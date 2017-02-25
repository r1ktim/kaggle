import numpy as np
from network import Network
import sys
import time

maxEpochs = 10000;
batch_size = 256;

lr = 0.001;
minLr = 0.00001;
decay = 0.99;
begin = time.time();
for e in range(maxEpochs):
    #print time.time() - begin;
    #begin = time.time();
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
            print 'Learning rate:', lr;
            testEval = [];
            
            for t in range(0, len(testInput), batch_size):
                if (t+batch_size < len(testInput)):
                    batchInput = testInput[t:t+batch_size];
                    batchTarget = testTarget[t:t+batch_size];    
                else:
                    batchInput = testInput[t:t + abs((abs(len(testInput) -(t + batch_size)) - batch_size))]
                    batchTarget = testTarget[t:t + abs((abs(len(testTarget) -(t + batch_size)) - batch_size))];
                    
                testEval.append(network.Eval(batchInput, batchTarget));
            
            network.Save("/home/rik/models/", str(e));
        
        
        print "Epoch:", e;
        print "Train correct:", np.mean(trainEval);
        if len(valEval) > 0:
            print "Val correct:",  np.mean(valEval);
        if (e % 100 == 0):
            print "Test correct:", np.mean(testEval);
        
    for t in range(0, len(trainInput), batch_size):
        if (t+batch_size < len(trainInput)):
            batchInput = trainInput[t:t+batch_size];
            batchTarget = trainTarget[t:t+batch_size];

        else:
            batchInput = trainInput[t:t + abs((abs(len(trainInput) -(t + batch_size)) - batch_size))]
            batchTarget = trainTarget[t:t + abs((abs(len(trainTarget) -(t + batch_size)) - batch_size))];

        network.Train(batchInput, batchTarget, lr);

    if lr >= minLr:
        lr = lr * decay;
    

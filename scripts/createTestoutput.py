import pandas as pd
import numpy as np
from topLayer import Network

x = np.load("/home/rik/KaggleData/mnist/featuresFinalTest.npy");

network = Network(1);
network.CreateNetwork();

network.Load("/home/rik/models/", "top_model800.ckpt");

nrOfExamples = len(x);


test_labels = [];
for e in range(nrOfExamples):
    output = network.Run([x[e]])[0];
    #print 'Input:', x[e];
    #print output
    test_labels.append(np.argmax(output));

test_labels = np.asarray(test_labels);
submission = pd.DataFrame(data={'ImageId':(np.arange(test_labels.shape[0])+1), 'Label':test_labels})
submission.to_csv('submission.csv', index=False)
submission.head()

print "Done";


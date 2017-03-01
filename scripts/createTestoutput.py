import pandas as pd
import numpy as np
from network import Network
import cv2
data = np.load("/home/rik/KaggleData/dogsVsCats/test.npy");

input = [];
labels = [];
for i in range(len(data)):
    input.append(data[i][0]);
    labels.append(data[i][1]);

network = Network(1);
network.CreateNetwork();

network.Load("/home/rik/models/dogsVsCats", "cnn_model280.ckpt");

nrOfExamples = len(input);

test_id = [];
test_labels = [];
for e in range(nrOfExamples):
    output = network.Run([input[e]])[0];
    '''
    img = input[e];
    img = cv2.resize(img, (320,320));
    
    print output[0];
    cv2.imshow("image", img);
    cv2.waitKey(0);
    '''
    #print 'Input:', x[e];
    #print output
    test_id.append(labels[e]);
    if (output[0] == 1):
        output[0] -= 0.00001;
    test_labels.append(output[0]);

test_labels = np.asarray(test_labels);
submission = pd.DataFrame(data={'id':test_id, 'label':test_labels})
submission.to_csv('submission.csv', index=False)
submission.head()

print "Done";


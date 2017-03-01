import numpy as np
import random

a = [1,2,3,4,5,6,7,8,9,10];
a = np.asarray(a);
index = range(len(a));
print index
random.shuffle(index);



c = a[index];
c = np.asarray(c);

print c
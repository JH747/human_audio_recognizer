import numpy as np

def labeller(length, val):
    tmp = np.full((length,1), val)
    return tmp


abc = labeller(3, 0.5)

print(abc)
print(abc.shape)

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

def parser(filename):
    f = open(filename, 'r')
    arr_2d = []

    while True:
        line = f.readline()
        if not line: break
        list1 = line.split(',')
        l1 = len(list1[0])
        l2 = len(list1[4871])
        list1[0] = list1[0][1:l1]
        list1[4871] = list1[4871][0:l2 - 2]
        for i in range (4872):
            list1[i] = float(list1[i])
        # print(list1[0], list1[4871])
        arr_2d.append(list1)
    tmp_npy = np.array(arr_2d)
    f.close()
    return tmp_npy


def labeller(length, val):
    tmp = np.full((length,1), val)
    return tmp


x_test_1d = parser("speech.txt")
y_test = labeller(90, 100)
print(x_test_1d.shape)
print(y_test.shape)

# ------------------------------- model start here -----------------------------

model = load_model('m2.h5')

y_predicted_result = model.predict(x_test_1d)

print(y_predicted_result)
print(y_predicted_result.shape)


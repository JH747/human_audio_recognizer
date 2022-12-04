import numpy as np
from keras.models import load_model

def parser(filename):
    f = open(filename, 'r')
    arr_2d = []

    while True:
        line = f.readline()
        if not line: break
        list1 = line.split(',')
        l1 = len(list1[0])
        l2 = len(list1[2435])
        list1[0] = list1[0][1:l1]
        list1[2435] = list1[2435][0:l2 - 2]
        for i in range (2436):
            list1[i] = float(list1[i])
        # print(list1[0], list1[2435])
        arr_2d.append(list1)
    tmp_npy = np.array(arr_2d)
    f.close()
    return tmp_npy

def labeller(length, val):
    tmp = np.full((length,1), val)
    return tmp

# ----------------------------- data load and prepare -----------------------------

x_train_1d = parser("drive/MyDrive/Colab Notebooks/file_name.txt")
y_train = labeller(90, 0)
print(x_train_1d.shape)
print(y_train.shape)

# ------------------------------- model start here -----------------------------

model = load_model('m1.h5')

hist = model.fit(x_train_1d, y_train, batch_size=16, epochs=100, shuffle=True) # shuffle=True

# y_predicted_result = model.predict(x_test)

# print(y_predicted_result)
# print(y_predicted_result.shape)

model.save("m2.h5")



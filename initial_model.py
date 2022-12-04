import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# ----------------------------- data parser function  -----------------------------

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

x1_train = parser("drive/MyDrive/Colab Notebooks/speech-0.txt")
x2_train = parser("drive/MyDrive/Colab Notebooks/speech-0(2).txt")
x3_train = parser("drive/MyDrive/Colab Notebooks/speech-0(demo).txt")
x4_train = parser("drive/MyDrive/Colab Notebooks/speech-1.txt")

y1_train = labeller(193, 0)
y2_train = labeller(209, 0)
y3_train = labeller(61, 0)
y4_train = labeller(501, 100)

print(x1_train.shape)
print(x2_train.shape)
print(x3_train.shape)
print(x4_train.shape)

xmute2 = np.concatenate((x1_train,x1_train),axis=0)
xmute3 = np.concatenate((xmute2,x1_train),axis=0) #
ymute2 = np.concatenate((y1_train,y1_train),axis=0)
ymute3 = np.concatenate((ymute2,y1_train),axis=0) #

xnoise2 = np.concatenate((x2_train,x2_train),axis=0) #
ynoise2 = np.concatenate((y2_train,y2_train),axis=0) #

xclap2 = np.concatenate((x3_train,x3_train),axis=0)
xclap4 = np.concatenate((xclap2,xclap2),axis=0)
xclap8 = np.concatenate((xclap4,xclap4),axis=0) #
yclap2 = np.concatenate((y3_train,y3_train),axis=0)
yclap4 = np.concatenate((yclap2,yclap2),axis=0)
yclap8 = np.concatenate((yclap4,yclap4),axis=0) #

xvoice2 = np.concatenate((x4_train,x4_train),axis=0) #
yvoice2 = np.concatenate((y4_train,y4_train),axis=0) #

# arr = np.concatenate((x1_train,x2_train),axis=0)
# x_train_1d = np.concatenate((arr,x3_train),axis=0)
# arr = np.concatenate((y1_train,y2_train),axis=0)
# y_train = np.concatenate((arr,y3_train),axis=0)

arr = np.concatenate((xmute3,xnoise2),axis=0)
arr2 = np.concatenate((arr,xclap8),axis=0)
x_train_1d = np.concatenate((arr2,xvoice2),axis=0)

arr = np.concatenate((ymute3,ynoise2),axis=0)
arr2 = np.concatenate((arr,yclap8),axis=0)
y_train = np.concatenate((arr2,yvoice2),axis=0)

idx = np.arange(x_train_1d.shape[0])
np.random.shuffle(idx)

x_train_1d = x_train_1d[idx]
y_train = y_train[idx]

# x_train_1d = x1_train + x2_train + x3_train
# y_train = y1_train + y2_train + y3_train
# print(arr.shape)
print(x_train_1d.shape)
print(y_train.shape)

# ------------------------------- model start here ---------------------------------

width = 21
height = 116

model = Sequential()
model.add(Dense(2436, activation='relu', input_dim = width*height))
# model.add(Dense(2436, activation='relu'))
# model.add(Dense(2436, activation='relu'))
model.add(Dense(1218, activation='relu'))
model.add(Dense(1))

model.compile(loss= 'mse', optimizer='adam') # metrics=['accuracy'] will result 0 because it will never match the exact number
hist = model.fit(x_train_1d, y_train, batch_size=16, epochs=50, shuffle=True) # shuffle=True

y_predicted_result = model.predict(x1_train)
y_predicted_result2 = model.predict(x4_train)

print(y_predicted_result)
print(y_predicted_result.shape)
print(y_predicted_result2)
print(y_predicted_result2.shape)

model.save("drive/MyDrive/Colab Notebooks/m1.h5")



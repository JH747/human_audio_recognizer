import numpy as np
from keras.models import Sequential
from keras.layers import Dense

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


x1_train = parser("drive/MyDrive/Colab Notebooks/speech.txt")
x2_train = parser("drive/MyDrive/Colab Notebooks/non-speech.txt")
x3_train = parser("drive/MyDrive/Colab Notebooks/mute.txt")

y1_train = labeller(90, 100)
y2_train = labeller(90, 0)
y3_train = labeller(90, 0)

print(x1_train.shape)

arr = np.concatenate((x1_train,x2_train),axis=0)
x_train_1d = np.concatenate((arr,x3_train),axis=0)
arr = np.concatenate((y1_train,y2_train),axis=0)
y_train = np.concatenate((arr,y3_train),axis=0)

idx = np.arange(x_train_1d.shape[0])
np.random.shuffle(idx)

x_train_1d = x_train_1d[idx]
y_train = y_train[idx]

print(x_train_1d.shape)
print(y_train.shape)

# ------------------------------- model start here ----------------------------- 

width = 21
height = 232

model = Sequential()
model.add(Dense(4872, activation='relu', input_dim = width*height))
model.add(Dense(2436, activation='relu'))
# model.add(Dense(2436, activation='relu'))
# model.add(Dense(1218, activation='relu'))
model.add(Dense(1))

model.compile(loss= 'mse', optimizer='adam') # metrics=['accuracy'] will result 0 because it will never match the exact number
hist = model.fit(x_train_1d, y_train, batch_size=16, epochs=100, shuffle=True) # shuffle=True

y_predicted_result = model.predict(x2_train)

print(y_predicted_result)
print(y_predicted_result.shape)

model.save("m1.h5")



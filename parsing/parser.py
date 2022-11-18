import numpy as np

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
    print(tmp_npy.shape)

    f.close()
    return tmp_npy

parser("non-speech.txt")


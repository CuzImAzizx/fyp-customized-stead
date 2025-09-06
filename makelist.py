import numpy as np

train = list(open("Anomaly_Train.txt"))
test = list(open("Anomaly_Test.txt"))

with open('ucf_x3d_train.txt', 'w+') as f:
    normal_files = []
    for file in train:
        if "Normal" in file:
            normal_files.append(file)
        else:
            newline = 'X3D_Videos/' + file[:-4] + 'npy\n'
            f.write(newline)
    for file in normal_files:
        newline = 'X3D_Videos/' + file[:-4] + 'npy\n'
        f.write(newline)

with open('ucf_x3d_test.txt', 'w+') as f:
    normal_files = []
    for file in test:
        if "Normal" in file:
            normal_files.append(file)
        else:
            newline = 'X3D_Videos/' + file[:-4] + 'npy\n'
            f.write(newline)
    for file in normal_files:
        newline = 'X3D_Videos/' + file[:-4] + 'npy\n'
        f.write(newline)

import matplotlib.pyplot as plt
import numpy as np

data = open("history_3.txt").readlines()

data = [d.strip().split(",") for d in data]
data = [[dd.split("=")[1] for dd in d] for d in data]

data = np.array(data)

history = {
    "train_acc" : data[:,0],
    "train_loss" : data[:,1],
    "val_acc" : data[:,2],
    "val_loss" : data[:,3]
}

plt.plot(history['train_acc'])
plt.ylabel('some numbers')

plt.show()
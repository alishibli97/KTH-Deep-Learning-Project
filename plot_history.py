import matplotlib.pyplot as plt
import numpy as np
from numpy.core.defchararray import index

data = open("results/history_18.txt").readlines()

data = [d.strip().split(",") for d in data]
data = [[float(dd.split("=")[1]) for dd in d] for d in data]

data = np.array(data)

history = {
    "train_acc" : data[:,0],
    "train_loss" : data[:,1],
    "val_acc" : data[:,2],
    "val_loss" : data[:,3]
}

train_acc = history['train_acc']
val_acc = history['val_acc']
train_loss = history['train_loss']
val_loss = history['val_loss']

sampling_rate = 32

interval = np.array(list(range(len(train_acc))))
filter_indexes = [i for i in range(len(interval)) if i%sampling_rate==0]
interval = interval[filter_indexes]
train_acc = train_acc[filter_indexes]
val_acc = val_acc[filter_indexes]

interval = np.array(list(range(len(train_loss))))
filter_indexes = [i for i in range(len(interval)) if i%sampling_rate==0]
interval = interval[filter_indexes]
train_loss = train_loss[filter_indexes]
val_loss = val_loss[filter_indexes]

fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('Accuracy and loss evolution')

ax1.plot(interval,train_acc,label="train")
ax1.plot(interval,val_acc,label="val")
ax1.legend()
ax1.set(xlabel='Iteration', ylabel='Accuracy')

ax2.plot(interval,train_loss,label="train")
ax2.plot(interval,val_loss,label="val")
ax2.legend()
ax2.set(xlabel='Iteration', ylabel='Loss')

fig.tight_layout(rect=[0, 0.03, 1, 0.95])

name = "Epoch_18_sampling_32.png"
plt.savefig(f"results/{name}")

plt.show()
import numpy as np
import matplotlib.pyplot as plt
import sys
import firstMLP as f

import numpy as np

data = np.array([[0,0],[0,1],[1,0],[1,1]])
labels = [
    np.array([0,0,0,1]),
    np.array([0,1,1,1]),
    np.array([0,1,1,0]),
    np.array([1,1,1,0]),
    np.array([1,0,0,0])
]



myMlp = f.MLP(2)
log = []


def epoch(mlp, labels):
    epochlogLoss = np.array([])
    epochlogAcc = np.array([])

    for i in range(4):
        input = data[i]
        target = labels[i]
        mlp.forward_step(input)
        mlp.backprop_step(target)
        loss = np.append(epochlogLoss,(target-mlp.prediction)**2)
        acc = np.append(epochlogAcc,np.abs(target-mlp.prediction)<0.5)

    log.append([loss.mean(),acc.mean()])
for i in range(1000):
    epoch(myMlp,labels[3])

plt.plot([l[0] for l in log],"-")
plt.plot([l[1] for l in log],".")

plt.show()

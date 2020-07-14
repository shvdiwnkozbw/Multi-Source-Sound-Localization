import numpy as np
from sklearn.metrics import auc

thres = 0.001

def cal_ciou(location, gtmap, thres):
    assert location.shape == gtmap.shape
    ciou = np.sum(gtmap[location>thres]) / (np.sum(gtmap)+np.sum(gtmap[location>thres]==0))
    return ciou

cious = []

gtmaps = np.load('gtmap.npy')
locations = np.load('location.npy')

for i in range(len(gtmaps)):
    ciou = cal_ciou(locations[i], gtmaps[i], thres)
    cious.append(ciou)

results = []
for i in range(21):
    result = np.sum(np.array(cious) >= 0.05 * i)
    result = result / len(cious)
    results.append(result)
x = [0.05 * i for i in range(21)]
auc = auc(x, results)
print(auc, results[10])

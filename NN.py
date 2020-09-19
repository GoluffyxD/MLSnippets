# Calculate weights for sampling
from sklearn.utils import class_weight
cw = class_weight.compute_class_weight('balanced',np.unique(y_train),y_train)
cls_wt = {}
for i in range(len(cw)):
    cls_wt[i] = cw[i]
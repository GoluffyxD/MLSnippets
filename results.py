import numpy as np
# Confusion Matrix

def printResults(confMatrix):
    # confMatrix = metrics.confusion_matrix(y_test,y_pred)
    # print(confMatrix)
    FP = confMatrix.sum(axis=0) - np.diag(confMatrix)  
    FN = confMatrix.sum(axis=1) - np.diag(confMatrix)
    TP = np.diag(confMatrix)
    TN = confMatrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    print('Sensitivity:',TPR)
    print('Specificity:',TNR)
    print('Precision:',PPV)
    print('Negative predictive value',NPV)
    print('False positive rate',FPR)
    print('False negative rate',FNR)
    print('False Discovery Rate',FDR)
    print('Overall Accuracy',ACC)
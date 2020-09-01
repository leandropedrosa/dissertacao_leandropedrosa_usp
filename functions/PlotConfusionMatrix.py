import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import itertools

# Função para traçar as matrizes de confusão
def plot_confusion_matrix(cm, title="Matriz de confusão"):
    classes=["AML", "ALL"]    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.bone)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    plt.ylabel('Real')
    plt.xlabel('Predito')
    thresh = cm.mean()
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]), 
                 horizontalalignment="center",
                 color="white" if cm[i, j] < thresh else "black") 
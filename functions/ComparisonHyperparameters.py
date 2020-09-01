import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from rdkit import Chem
from math import floor
from time import time
import pylab as pl
import itertools
import matplotlib.pyplot as plt 
from functions.Hyperparameter import procurar
from functions.PlotConfusionMatrix import plot_confusion_matrix
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss

# PROCURAR POR PARAMETROS
def compare(estimator, X_train, Y_train, param, dist, descritor, algoritimo):
    grid = procurar(estimator, X_train, Y_train, param, "grid")
    cfmatrix_grid = confusion_matrix(y_true=Y_train, y_pred=grid.predict(X_train))
    print("**Resultados Grid Search**")
    print("Melhores Parâmetros: %s" % (grid.best_params_))
    print("Melhor precisão de treinamento:\t", grid.best_score_)

    random = procurar(estimator, X_train, Y_train, dist, "random")
    cfmatrix_rand = confusion_matrix(y_true=Y_train, y_pred=random.predict(X_train))
    print("**Resultados Random Search**")
    print("Melhores Parâmetros: %s" % (random.best_params_))
    print("Melhor precisão de treinamento:\t", random.best_score_)

    plt.subplots(1,2)
    plt.subplots_adjust(left=-0.5, bottom=None, right=None, top=None, wspace=0.5, hspace=None)    
    plot_confusion_matrix(cfmatrix_rand, title="RandomSearchX"+descritor+'X'+algoritimo)
    plt.subplot(121)
    plot_confusion_matrix(cfmatrix_grid, title="GridSearchX"+descritor+'X'+algoritimo)
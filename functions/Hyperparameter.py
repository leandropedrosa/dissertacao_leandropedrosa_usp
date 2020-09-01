import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from rdkit import Chem
from math import floor
from time import time
import pylab as pl

# PROCURAR POR PARAMETROS
def procurar(estimator, X_train, Y_train, param_grid, search):
    try:
        if search == "grid":
            clf = GridSearchCV(
                estimator=estimator, 
                param_grid=param_grid, 
                scoring=None,
                n_jobs=-1, 
                cv=5, 
                verbose=5,
                return_train_score=True
            )
        elif search == "random":           
            clf = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=param_grid,
                n_iter=45,
                n_jobs=-1,
                cv=5,
                verbose=5,
                random_state=1,
                return_train_score=True
            )
    except:
        print('O argumento de pesquisa deve ser "grade" ou "aleatório"')
        sys.exit(0)
        
    # Ajuste o modelo
    clf.fit(X=X_train, y=Y_train)
    
    return clf 

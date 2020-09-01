import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from rdkit import Chem
from math import floor
from time import time
import pylab as pl

## Construção de modelo

##### Função utilitária para relatar as melhores pontuações
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Modelo com classificação: {0}".format(i))
            print("Escore médio de validação: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parametros: {0}".format(results['params'][candidate]))
            print("")
            
### Pesquisa em grade (Grid Search)       
def grid_search(classificador, grid_params, X_train, y_train):
    # configuração detalhada = 10 imprimirá o progresso para cada 10 tarefas concluídas
    grid_search = GridSearchCV(classificador, grid_params, n_jobs=-1, cv=5, verbose=1)
    grid_search.fit(X_train, y_train)

    return grid_search

### Pesquisa aleatória (Random Search)
def random_search(classificador, param_dist, X_train, y_train):
    n_iter_search = 110
    random_search = RandomizedSearchCV(classificador, param_distributions=param_dist, n_iter=n_iter_search, n_jobs=-1, cv=5, verbose=1)
    random_search.fit(X_train, y_train)
    
    return random_search

### Pesquisa em grade (Grid Search) 
def best_grid_search(classificador, param_grid, X_train, y_train, descritor, algoritimo):
    # Grid Search
    grid = grid_search(classificador, param_grid, X_train, y_train)
    grid.fit(X_train, y_train)
    print('Grid Search Best params %s X %s: %s' % (descritor,grid.best_params_))
    print('Score: %.2f' % grid.best_score_)
    
    return grid

### Pesquisa em grade (Grid Search) 
def best_random_search(classificador, param_grid, X_train, y_train, descritor, algoritimo):
    # Random Search
    random = random_search(classificador, param_grid, X_train, y_train)
    random.fit(X_train, y_train)
    print('Random Search Best params %s X %s: %s' % (descritor, algoritimo,random.best_params_))
    print('Score: %.2f' % random.best_score_)
    
    return random

from sklearn.model_selection import permutation_test_score
import pandas as pd
import numpy as np
import pylab as pl
from rdkit import Chem
from math import floor

# Avaliar a significância de uma pontuação validada cruzada com permutações
# True score = A pontuação verdadeira, sem permutar metas.
# Y-randomization = Media das pontuações obtidas para cada permutação.
# np-value = O valor retornado é igual ao valor-p se a pontuação retornar números maiores para obter melhores 
#     pontuações (por exemplo, exatidão_score). Se a pontuação é uma função de perda (ou seja, quando menor é melhor, 
#     como com mean_squared_error), então esse é realmente o complemento do valor p: 1 - valor p.
##### Função utilitária para relatar as melhores pontuações
def y_randomization(rf_best, X_train, y_train, descritor, algoritimo):    
    permutations = 20
    score, permutation_scores, pvalue = permutation_test_score(rf_best, X_train, y_train,
                                                               cv=5, scoring='balanced_accuracy',
                                                               n_permutations=permutations,
                                                               n_jobs=-1,
                                                               verbose=1,
                                                               random_state=24)
    print('True score = ', score.round(2),
          '\n Média per. = ', np.mean(permutation_scores).round(2),
          '\np-value = ', pvalue.round(4))

    ###############################################################################
    # View histogram of permutation scores
    pl.subplots(figsize=(10,6))
    pl.hist(permutation_scores.round(2), label='Permutation scores')
    ylim = pl.ylim()
    pl.vlines(score, ylim[0], ylim[1], linestyle='--',
              color='g', linewidth=3, label='Classification Score'
              ' (pvalue %s)' % pvalue.round(4))
    pl.vlines(1.0 / 2, ylim[0], ylim[1], linestyle='--',
              color='k', linewidth=3, label='Luck')
    pl.ylim(ylim)
    pl.legend()
    pl.xlabel('Score')
    pl.title('Aleatoriarização da variável Y '+algoritimo+'X'+descritor, fontsize=12)
    pl.savefig('figures/y_randomization-'+descritor+'X'+algoritimo+'.png', bbox_inches='tight', transparent=False, format='png', dpi=300)
    pl.show()
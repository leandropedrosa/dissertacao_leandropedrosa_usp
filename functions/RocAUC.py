import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from math import floor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_roc_curve, auc


def roc_auc(rf_best, cross_val, X_train, y_train, descritor, algoritimo):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(10,6))
    for i, (train_index, test_index) in enumerate(cross_val.split(X_train, y_train)):
        rf_best.fit(X_train.iloc[train_index], y_train[train_index])
        viz = plot_roc_curve(rf_best, X_train.iloc[test_index], y_train[test_index],
                             name='ROC fold {}'.format(i),
                             alpha=0.3, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8, )

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Média ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="AUC - Descritor: "+descritor+" - Algoritmo: "+algoritimo)
    ax.legend(loc="lower right")
    plt.savefig('figures/auc-5f-'+descritor+'.png', bbox_inches='tight', transparent=False, format='png', dpi=300)
    plt.show()

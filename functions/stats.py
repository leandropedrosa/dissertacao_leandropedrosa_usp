import pandas as pd
from sklearn import metrics

def stats(y_train, y_pred):
    confusion_matrix = metrics.confusion_matrix(y_train, y_pred, labels=[0,1])
    Kappa = metrics.cohen_kappa_score(y_train, y_pred, weights='linear')
    # Valores verdadeiros e falsos
    TN, FP, FN, TP = confusion_matrix.ravel()
    # Accuracy
    AC = (TP+TN)/(TP+FP+FN+TN)
    # Sensibilidade, taxa de acerto, recall ou taxa positiva verdadeira
    SE = TP/(TP+FN)
    # Especificidade ou taxa negativa verdadeira
    SP = TN/(TN+FP)
    # Precisão ou valor preditivo positivo
    PPV = TP/(TP+FP)
    # Valor preditivo negativo
    NPV = TN/(TN+FN)
    # Taxa de classificação correta
    CCR = (SE + SP)/2   
    # F1 Score
    F1_score = 2*(PPV*SE)/(PPV+SE)
    d = dict({'Kappa': Kappa,
         'AUC': CCR,
         'Sensibilidade': SE,
         'PPV': PPV,
         'Especificidade': SP,
         'NPV': NPV,
         'Acurácia': AC,
         'F1 Score':F1_score})
    return pd.DataFrame(d, columns=d.keys(), index=[0]).round(2)

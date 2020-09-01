import numpy as np
import pandas as pd
from functions.stats import *


def statistics(rf_best, X_train, y_train, cross_val, moldf_desc, moldf_train, nome):
    # Parametros
    pred = []
    ad = []
    index = []

    # Faça um loop de 5 vezes
    for train_index, test_index in cross_val.split(X_train, y_train):    
        fold_model = rf_best.fit(X_train.iloc[train_index], y_train[train_index])
        fold_pred = rf_best.predict(X_train.iloc[test_index])
        fold_ad = rf_best.predict_proba(X_train.iloc[test_index])
        pred.append(fold_pred)
        ad.append(fold_ad)
        index.append(test_index)

    # nomal = média majoritária das previsões dos modelos independentes desenvolvidos com os decritores de descritor
    # AD = previsões médias de modelos independentes quando as previsões estão dentro do domínio de aplicabilidade desse modelo

    threshold_ad = 0.70 # A abordagem do domínio de aplicabilidade local (árvore), que estabeleceu um limiar de 70%, foi utilizada para todos os modelos de RF

    # Preparar resultados para exportar    
    fold_index = np.concatenate(index)    
    fold_pred = np.concatenate(pred)
    fold_ad = np.concatenate(ad)
    # amax() retorna o máximo de uma matriz ou o máximo ao longo do eixo (se mencionado).
    # astype() também oferece a capacidade de converter qualquer coluna existente adequada em tipo categórico.
    fold_ad = (np.amax(fold_ad, axis=1) >= threshold_ad).astype(str)
    five_fold_descritor = pd.DataFrame({'Prediction': fold_pred,'AD': fold_ad}, index=list(fold_index))
    five_fold_descritor.AD[five_fold_descritor.AD == 'False'] = np.nan
    five_fold_descritor.AD[five_fold_descritor.AD == 'True'] = five_fold_descritor.Prediction
    five_fold_descritor.sort_index(inplace=True)
    five_fold_descritor['y_train'] = pd.DataFrame(y_train)
    five_fold_ad = five_fold_descritor.dropna().astype(int)
    coverage_5f = len(five_fold_ad) / len(five_fold_descritor)


    # estatísticas de descritor (stats)
    descritor = pd.DataFrame(stats(five_fold_descritor['y_train'], five_fold_descritor['Prediction']))
    descritor['Coverage'] = 1.0

    # estatísticas do descritor AD
    descritor_ad = five_fold_descritor.dropna(subset=['AD']).astype(int)
    coverage_descritor_ad = len(descritor_ad['AD']) / len(five_fold_descritor['y_train'])
    descritor_ad = pd.DataFrame(stats(descritor_ad['y_train'], descritor_ad['AD']))
    descritor_ad['Coverage'] = round(coverage_descritor_ad, 2)

    # imprimir estatísticas
    descritor_5f_stats = descritor.append(descritor_ad)
    descritor_5f_stats.set_index([[nome, nome+' AD']], drop=True, inplace=True)
    descritor_5f_stats

    # Valores verdadeiros e falsos: TN, FP, FN, TP = confusion_matrix.ravel()
    # Accuracy AC = (TP+TN)/(TP+FP+FN+TN)
    # Sensibilidade, taxa de acerto, recall ou taxa positiva verdadeira: SE = TP/(TP+FN)
    # Especificidade ou taxa negativa verdadeira: SP = TN/(TN+FP)
    # Precisão ou valor preditivo positivo: PPV = TP/(TP+FP)
    # Valor preditivo negativo: NPV = TN/(TN+FN)
    # Taxa de classificação correta: CCR = (SE + SP)/2
    # Cobertura: baseia-se na cobertura média (número de amostras afetadas) de divisões que usam o recurso
    # F1 Score = 2 * (Precision * Sensitivity) / (Precision + Sensitivity)
    # O índice de acurácia Kappa, é uma pontuação que expressa o nível de concordância entre dois anotadores em um problema de classificação. É definido como k = (Po-Pc)/(1-Pc) (y_train, y_pred, weights='linear')
    # Po = Precisão Global (Proporção de unidades que concordam); e
    # Pc = Proporção de unidades que concordam por coincidência, representada pela Equação:
    # Pc = Somatoria(M)*ni+n+i/N^2
    # M = número de classes;
    # ni+ = total de elementos classificados para categoria i;
    # n+i = total de elementos de referência amostrados para uma categoria i; e
    # N = número total de amostras.
    # Tipo de ponderação para calcular a pontuação. Nenhum significa não ponderado; "Linear" significa linear ponderado; "Quadrático" significa ponderado quadrático.

    ##### Prever conjunto retido externo após o balanceamento
    moldf_ext = moldf_desc[(moldf_desc['Set'] == 'ext')]
    descriptor_list = list(X_train.columns.values)

    if len(moldf_ext) > 0:
        y_ext = moldf_ext['Outcome'].to_numpy()
        X_ext = moldf_ext[descriptors]

        # Filtrar descritores não presentes no modelo
        X_ext = X_ext[descriptor_list]

        # Fazer previsões
        ext_set = rf_best.predict(X_ext)
        ext_set_ad = rf_best.predict_proba(X_ext)
        ext_set_ad = (np.amax(ext_set_ad, axis=1) >= threshold_ad).astype(str)

        # Preparar dados
        ext_set = pd.DataFrame({'Prediction': ext_set,'AD': ext_set_ad})
        ext_set.AD[ext_set.AD == 'False'] = np.nan
        ext_set.AD[ext_set.AD == 'True'] = ext_set.Prediction
        ext_set.sort_index(inplace=True)
        ext_set['y_ext'] = pd.DataFrame(y_ext)
        ext_set_ad = ext_set.dropna().astype(int)
        coverage_ext = len(ext_set_ad) / len(ext_set)

        # Imprimir estatísticas
        ext_set_stats = pd.DataFrame(stats(ext_set.y_ext, ext_set.Prediction))
        ext_set_stats['Coverage'] = 1.0
        print('External withheld set: \n', ext_set_stats.to_string(index=False), '\n')
        ext_set_stats_ad = pd.DataFrame(stats(ext_set_ad.y_ext, ext_set_ad.AD))
        ext_set_stats_ad['Coverage'] = round(coverage_ext, 2)
        print('External withheld set with AD: \n', ext_set_stats_ad.to_string(index=False), '\n')
    else:
        pass

    pred_train = moldf_train[moldf_train.columns.difference(descriptor_list)]
    pred_train.reset_index(inplace=True)
    five_fold_exp = five_fold_descritor[['Prediction', 'AD']].rename(columns={'Prediction':'descritor', 'AD':'descritor_ad'}, )
    pred_train = pd.concat([pred_train, five_fold_exp], axis=1)
    pred_train['Set'] = 'train'

    if len(moldf_ext) > 0:
        pred_ext = moldf_ext[moldf_ext.columns.difference(descriptor_list)]
        pred_ext.reset_index(inplace=True)
        ext_set_exp = ext_set[['Prediction', 'AD']].rename(columns={'Prediction':'descritor', 'AD':'descritor_ad'}, )
        pred_ext = pd.concat([pred_ext, ext_set_exp], axis=1)
        pred_ext['Set'] = 'ext'

        pred_descritor = pd.concat([pred_train, pred_ext], axis=0).sort_values(by='index')
        pred_descritor.drop(columns=['index', 'Mol', 'ID'], inplace=True)

    else:
        pred_descritor = pred_train.copy()
        pred_descritor.drop(columns=['index', 'Mol', 'ID'], inplace=True)

    ### Exportar previsões
    pred_train = moldf_train[moldf_train.columns.difference(descriptor_list)]
    pred_train.reset_index(inplace=True)
    five_fold_exp = five_fold_descritor[['Prediction', 'AD']].rename(columns={'Prediction':'descritor', 'AD':'descritor_ad'}, )
    pred_train = pd.concat([pred_train, five_fold_exp], axis=1)
    pred_train['Set'] = 'train'

    if len(moldf_ext) > 0:
        pred_ext = moldf_ext[moldf_ext.columns.difference(descriptor_list)]
        pred_ext.reset_index(inplace=True)
        ext_set_exp = ext_set[['Prediction', 'AD']].rename(columns={'Prediction':'descritor', 'AD':'descritor_ad'}, )
        pred_ext = pd.concat([pred_ext, ext_set_exp], axis=1)
        pred_ext['Set'] = 'ext'

        pred_descritor = pd.concat([pred_train, pred_ext], axis=0).sort_values(by='index')
        pred_descritor.drop(columns=['index', 'Mol', 'ID'], inplace=True)

    else:
        pred_descritor = pred_train.copy()
        pred_descritor.drop(columns=['index', 'Mol', 'ID'], inplace=True)

    with pd.ExcelWriter('predictions/predictions-'+nome+'.xlsx') as writer:
        pred_descritor.to_excel(writer, sheet_name=nome, index=False)        
        
    ### Plotando as Estatísticas
    # Exportar estatísticas
    if len(moldf_ext) > 0:
        descritor_stats = pd.concat([descritor, descritor_ad], axis=0)
        descritor_stats.set_index([['5-fold CV', 'External withheld set']], drop=True, inplace=True)
        descritor_stats
    else:
        descritor_stats = descritor_5f_stats.copy()
        descritor_stats

    data_stats = {'five_fold': five_fold_descritor, 'stats': descritor_stats}
    return data_stats;
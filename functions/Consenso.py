import numpy as np
import pandas as pd
from functions.stats import stats

def statistics_morganXpadel(moldf, moldf_train, five_fold_morgan, five_fold_padel):
    
    results_morgan = five_fold_morgan.drop(columns='y_train')
    results_morgan = five_fold_morgan.rename(columns={'Prediction':'morgan', 'AD':'morgan_ad'})
    results_padel = five_fold_padel.drop(columns='y_train')
    results_padel = five_fold_padel.rename(columns={'Prediction':'padel', 'AD':'padel_ad'})
    
    var = list(moldf.columns.values)
    moldf_train = moldf_train[var]
    predictions = pd.concat([moldf_train.reset_index(drop=True), results_morgan, results_padel], axis=1)
    
    # Consensus
    predictions['consensus'] = (predictions.morgan + predictions.padel)/2
    predictions['consensus'] = np.where(predictions['consensus'] > 0.5, 1, 0)

    # Consensus AD
    for i in range(0, predictions.shape[0]):
        if all([np.isnan(predictions.morgan_ad[i]) == False, np.isnan(predictions.padel_ad[i]) == False]):
            predictions.loc[i,'consensus_ad'] = (predictions.morgan_ad[i] + predictions.padel_ad[i])/2
            predictions.loc[i,'consensus_ad'] = np.where(predictions.loc[i,'consensus_ad'] > 0.5, 1, 0)
        elif all([np.isnan(predictions.morgan_ad[i]) == True, np.isnan(predictions.padel_ad[i]) == False]):
            predictions.loc[i,'consensus_ad'] = predictions.padel_ad[i]
        elif all([np.isnan(predictions.morgan_ad[i]) == False, np.isnan(predictions.padel_ad[i]) == True]):
            predictions.loc[i,'consensus_ad'] = predictions.morgan_ad[i]
        else:
            predictions.loc[i,'consensus_ad']  = np.nan

    # Consensus Rigor
    for i in range(0, predictions.shape[0]):
        if all([np.isnan(predictions.morgan_ad[i]) == False, np.isnan(predictions.padel_ad[i]) == False]):
            predictions.loc[i,'consensus_rigor'] = (predictions.morgan_ad[i] + predictions.padel_ad[i])/2
            predictions.loc[i,'consensus_rigor'] = np.where(predictions.loc[i,'consensus_rigor'] > 0.5, 1, 0)
        else:
            predictions.loc[i,'consensus_rigor']  = np.nan

    predictions.drop(columns=['y_train', 'ID'], inplace=True)
    
    
    # morgan stats
    morgan = pd.DataFrame(stats(predictions.Outcome, predictions.morgan))
    morgan['Coverage'] = 1.0

    # morgan AD stats
    morgan_ad = predictions.dropna(subset=['morgan_ad'])
    coverage_morgan_ad = len(morgan_ad.morgan_ad) / len(predictions.Outcome)
    morgan_ad = pd.DataFrame(stats(morgan_ad.Outcome, morgan_ad.morgan_ad.astype(int)))
    morgan_ad['Coverage'] = round(coverage_morgan_ad, 2)

    ##### padel

    # padel stats
    padel = pd.DataFrame(stats(predictions.Outcome, predictions.padel))
    padel['Coverage'] = 1.0

    # padel AD stats
    padel_ad = predictions.dropna(subset=['padel_ad'])
    coverage_padel_ad = len(padel_ad.padel_ad) / len(predictions.Outcome)
    padel_ad = pd.DataFrame(stats(padel_ad.Outcome, padel_ad.padel_ad.astype(int)))
    padel_ad['Coverage'] = round(coverage_padel_ad, 2)

    ##### Consensus

    # consensus stats
    consensus = pd.DataFrame(stats(predictions.Outcome, predictions.consensus))
    consensus['Coverage'] = 1.0

    # consensus AD stats
    consensus_ad = predictions.dropna(subset=['consensus_ad'])
    coverage_consensus_ad = len(consensus_ad.consensus_ad) / len(predictions.Outcome)

    consensus_ad = pd.DataFrame(stats(consensus_ad.Outcome, consensus_ad.consensus_ad.astype(int)))
    consensus_ad['Coverage'] = round(coverage_consensus_ad, 2)

    # consensus rigor stats
    consensus_rigor = predictions.dropna(subset=['consensus_rigor'])
    coverage_consensus_rigor = len(consensus_rigor.consensus_rigor) / len(predictions.Outcome)
    consensus_rigor = pd.DataFrame(stats(consensus_rigor.Outcome, consensus_rigor.consensus_rigor.astype(int)))
    consensus_rigor['Coverage'] = round(coverage_consensus_rigor, 2)
    
    pred_exp = predictions.drop(columns=['Mol'])

    with pd.ExcelWriter('predictions/predictions-morgan-padel.xlsx') as writer:
        pred_exp.to_excel(writer, sheet_name='morgan-padel', index=False)

        
    stats_return = pd.concat([morgan_ad, padel_ad, consensus, consensus_ad, consensus_rigor], axis=0)
    stats_return.set_index([['Morgan', 'PaDEL', 'Consensus', 'Consensus (AD)', 'Consensus (Rigor)']], drop=True, inplace=True)
        
    return stats_return;

def statistics_sirmsXdragon(moldf, moldf_train, five_fold_sirms, five_fold_dragon):
    
   
    results_sirms = five_fold_sirms.drop(columns='y_train')
    results_sirms = five_fold_sirms.rename(columns={'Prediction':'sirms', 'AD':'sirms_ad'})
    results_dragon = five_fold_dragon.drop(columns='y_train')
    results_dragon = five_fold_dragon.rename(columns={'Prediction':'dragon', 'AD':'dragon_ad'})
    
    var = list(moldf.columns.values)
    moldf_train = moldf_train[var]
    predictions = pd.concat([moldf_train.reset_index(drop=True), results_sirms, results_dragon], axis=1)
    
    predictions['consensus'] = (predictions.sirms + predictions.dragon)/2
    predictions['consensus'] = np.where(predictions['consensus'] > 0.5, 1, 0)

    # Consensus AD
    for i in range(0, predictions.shape[0]):
        if all([np.isnan(predictions.sirms_ad[i]) == False, np.isnan(predictions.dragon_ad[i]) == False]):
            predictions.loc[i,'consensus_ad'] = (predictions.sirms_ad[i] + predictions.dragon_ad[i])/2
            predictions.loc[i,'consensus_ad'] = np.where(predictions.loc[i,'consensus_ad'] > 0.5, 1, 0)
        elif all([np.isnan(predictions.sirms_ad[i]) == True, np.isnan(predictions.dragon_ad[i]) == False]):
            predictions.loc[i,'consensus_ad'] = predictions.dragon_ad[i]
        elif all([np.isnan(predictions.sirms_ad[i]) == False, np.isnan(predictions.dragon_ad[i]) == True]):
            predictions.loc[i,'consensus_ad'] = predictions.sirms_ad[i]
        else:
            predictions.loc[i,'consensus_ad']  = np.nan

    # Consensus Rigor
    for i in range(0, predictions.shape[0]):
        if all([np.isnan(predictions.sirms_ad[i]) == False, np.isnan(predictions.dragon_ad[i]) == False]):
            predictions.loc[i,'consensus_rigor'] = (predictions.sirms_ad[i] + predictions.dragon_ad[i])/2
            predictions.loc[i,'consensus_rigor'] = np.where(predictions.loc[i,'consensus_rigor'] > 0.5, 1, 0)
        else:
            predictions.loc[i,'consensus_rigor']  = np.nan

    predictions.drop(columns=['y_train', 'ID'], inplace=True)
    
    
    ##### SiRMS

    # SiRMS stats
    sirms = pd.DataFrame(stats(predictions.Outcome, predictions.sirms))
    sirms['Coverage'] = 1.0

    # SiRMS AD stats
    sirms_ad = predictions.dropna(subset=['sirms_ad'])
    coverage_sirms_ad = len(sirms_ad.sirms_ad) / len(predictions.Outcome)
    sirms_ad = pd.DataFrame(stats(sirms_ad.Outcome, sirms_ad.sirms_ad.astype(int)))
    sirms_ad['Coverage'] = round(coverage_sirms_ad, 2)

    ##### DRAGON

    # dragon stats
    dragon = pd.DataFrame(stats(predictions.Outcome, predictions.dragon))
    dragon['Coverage'] = 1.0

    # dragon AD stats
    dragon_ad = predictions.dropna(subset=['dragon_ad'])
    coverage_dragon_ad = len(dragon_ad.dragon_ad) / len(predictions.Outcome)
    dragon_ad = pd.DataFrame(stats(dragon_ad.Outcome, dragon_ad.dragon_ad.astype(int)))
    dragon_ad['Coverage'] = round(coverage_dragon_ad, 2)

    ##### Consensus

    # consensus stats
    consensus = pd.DataFrame(stats(predictions.Outcome, predictions.consensus))
    consensus['Coverage'] = 1.0

    # consensus AD stats
    consensus_ad = predictions.dropna(subset=['consensus_ad'])
    coverage_consensus_ad = len(consensus_ad.consensus_ad) / len(predictions.Outcome)

    consensus_ad = pd.DataFrame(stats(consensus_ad.Outcome, consensus_ad.consensus_ad.astype(int)))
    consensus_ad['Coverage'] = round(coverage_consensus_ad, 2)

    # consensus rigor stats
    consensus_rigor = predictions.dropna(subset=['consensus_rigor'])
    coverage_consensus_rigor = len(consensus_rigor.consensus_rigor) / len(predictions.Outcome)
    consensus_rigor = pd.DataFrame(stats(consensus_rigor.Outcome, consensus_rigor.consensus_rigor.astype(int)))
    consensus_rigor['Coverage'] = round(coverage_consensus_rigor, 2)

    pred_exp = predictions.drop(columns=['Mol'])
    
    with pd.ExcelWriter('predictions/predictions-sirms-dragon.xlsx') as writer:
        pred_exp.to_excel(writer, sheet_name='sirms-dragon', index=False)

        
    stats_return = pd.concat([sirms_ad, dragon_ad, consensus, consensus_ad, consensus_rigor], axis=0)
    stats_return.set_index([['SiRMS', 'Dragon', 'Consensus', 'Consensus (AD)', 'Consensus (Rigor)']], drop=True, inplace=True)
        
    return stats_return;
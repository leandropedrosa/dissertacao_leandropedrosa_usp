import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from math import floor
#Rdkit: coleção de quiminformática e software de aprendizado de máquina escrito em C++ e Python de Código Aberto.
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import PandasTools
from rdkit.Chem.Draw import IPythonConsole
from collections import Counter
from sklearn.feature_selection import VarianceThreshold

def dragon_descriptors(moldf):   
    desc = pd.read_csv('descriptors/dragon-chembl-sars-cov-3C-like-proteinase-processed.txt', sep='\t')
    desc.drop(desc.columns[0:1], axis=1,inplace=True)
    descriptors = desc.columns.difference(moldf.columns).tolist()

    moldf_desc = pd.concat([moldf,desc], axis=1)
    moldf_desc['Set'] = 'train'
    moldf_train = moldf_desc[(moldf_desc['Set'] == 'train')]

    y_train = moldf_train['Outcome'].to_numpy()
    X_train = moldf_train[descriptors]
    X_train.shape
    
    data_train = {'moldf_desc': moldf_desc, 'moldf_train': moldf_train, 'Y_train': moldf_train['Outcome'].to_numpy(), 'X_train': moldf_train[descriptors]}
    return data_train
    
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

def morgan_descriptors(moldf):   
    ## Forma dos dados
    # (27 ativos e 64 inativos) 91 compostos utilizando o software ChemAxon Standardizer 
    # (13 ativos e 09 inativos) 22 compostos obtidos de empresas encontradas do PDB
    # 2 Classes criadas Classe 1: 40 (Ativos) e Classe 0: 73 (Inativos)

    moldf['Outcome'] = moldf['Outcome'].replace('Active', 1)
    moldf['Outcome'] = moldf['Outcome'].replace('Inactive', 0)

    classes = Counter(moldf['Outcome'])
    print('\033[1m' + 'Forma do conjunto de treinamento:' + '\n' + '\033[0m')
    for key, value in classes.items():
        print('\t\t Classe %d: %d' % (key, value))
    print('\t\t Número total de compostos: %d' % (len(moldf['Outcome'])))

    print('Class labels:', np.unique(classes))
    
    # Calculando os descritores fingerprints de Harry Morgan (vetores de bits).
    def calcfp(mol,funcFPInfo=dict(radius=3, nBits=2048, useFeatures=False, useChirality=False)):
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, **funcFPInfo)
        fp = pd.Series(np.asarray(fp))
        fp = fp.add_prefix('Bit_')
        return fp

    # Adicionando os 113 componentes e os 2048 dados referetens aos descritores de Morgan
    desc = moldf.Mol.apply(calcfp)
    descriptors = desc.columns.difference(moldf.columns).tolist()
    desc.shape
    
    # Moldando o conjunto de treinamento e o conjunto de validação externa
    moldf_desc = pd.concat([moldf,desc], axis=1)
    balance_data = 'no'

    if balance_data == 'yes':
        # Equilibre os dados usando 1/2 similaridade e 1/2 aleatória
        moldf_desc = BalanceBySim(moldf_desc, 'Outcome', 2)
        # Forma de impressão
        print('Forma do conjunto de treinamento: %s' % Counter(moldf_desc['Outcome'].loc[moldf_desc['Set'] == 'train']))
        print('Forma externa definida: %s' % Counter(moldf_desc['Outcome'].loc[moldf_desc['Set'] == 'ext']))

    else:
        moldf_desc['Set'] = 'train'
        # Forma de impressão
        print('Forma do conjunto de treinamento: %s' % Counter(moldf_desc['Outcome'].loc[moldf_desc['Set'] == 'train']))
        print('Forma externa definida: %s' % Counter(moldf_desc['Outcome'].loc[moldf_desc['Set'] == 'ext']))
    
    # Conjunto de treinamento
    moldf_train = moldf_desc[(moldf_desc['Set'] == 'train')]
    
    data_train = {'moldf_desc': moldf_desc, 'moldf_train': moldf_train, 'Y_train': moldf_train['Outcome'].to_numpy(), 'X_train': moldf_train[descriptors]}
    return data_train
    
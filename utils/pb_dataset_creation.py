import glob
from scipy.io.arff import loadarff 
import pandas as pd
import os
import pickle

# transform polish banks dataset
path = r'D:\everything\Studium\SS2023\ml_ss23_group13\data\Polish_Banks\\'
dfs = []
for file in glob.glob(f'{path}*'):
    raw_data = loadarff(file)
    dfs.append(pd.DataFrame(raw_data[0]))
df = pd.concat(dfs)

df.to_csv(f'{path}polish_banks_dataset.csv')

# create dict with names for polish banks dataset
f = open(f'{path}attr_names.txt', "r")
attr_names = f.read().split('\n')

attr_names_dict = {}
for attr, name in zip(df.columns, attr_names):
    attr_names_dict[attr] = name
attr_names_dict['class'] = 'class'

with open(f'{path}attr_names.pickle', 'wb') as handle:
    pickle.dump(attr_names_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
import pandas as pd
import numpy as np

df = pd.read_csv('C:/thesis-goettert/data/testData4.csv', sep = ',', quotechar = '"')
n = 0
to_delete = ['fis-','gesis-','dzi-','csa-','ubk-','iab-','OR OR','fes-','dza-','wzb-','proquest-']
delete_sessions = []
for index, row in df.iterrows():
    print(n/7982427)
    n += 1
    if row['params'] is not np.nan:
        if any(x in row['params'] for x in to_delete) & ('searchterm' in row['mapping_action_label']):
            delete_sessions.append(row['key'])
        row['params'] = row['params'].replace('ODER','OR')
        row['params'] = row['params'].replace('UND','AND')
        row['params'] = row['params'].replace('%20',' ')
        row['params'] = row['params'].replace('%21','!')
        row['params'] = row['params'].replace('%22','"')
        row['params'] = row['params'].replace('%23','#')
        row['params'] = row['params'].replace('%24','$')
        row['params'] = row['params'].replace('%25','%')
        row['params'] = row['params'].replace('%26','&')
        row['params'] = row['params'].replace('%27','\'')
        row['params'] = row['params'].replace('%28','(')
        row['params'] = row['params'].replace('%29',')')
        row['params'] = row['params'].replace('%2A','*')
        row['params'] = row['params'].replace('%2B','+')
        row['params'] = row['params'].replace('%2C',',')
        row['params'] = row['params'].replace('%2D','-')
        row['params'] = row['params'].replace('%2E','.')
        row['params'] = row['params'].replace('%2F','/')
        row['params'] = row['params'].replace('%3A',':')
        row['params'] = row['params'].replace('%3B',';')
        row['params'] = row['params'].replace('%3C','<')
        row['params'] = row['params'].replace('%3D','=')
        row['params'] = row['params'].replace('%3E','>')
        row['params'] = row['params'].replace('%3F','?')
        row['params'] = row['params'].replace('%40','@')
        row['params'] = row['params'].replace('%5B','[')
        row['params'] = row['params'].replace('%5C','\\')
        row['params'] = row['params'].replace('%5D',']')
        row['params'] = row['params'].replace('%7B','{')
        row['params'] = row['params'].replace('%7C','|')
        row['params'] = row['params'].replace('%7D','}')


        df.at[index, 'params'] = row['params']
df = df.set_index(['key'])
df = df[~df.index.isin(delete_sessions)]      

df.reset_index(level=df.index.names, inplace=True)
df.to_csv("C:/thesis-goettert/data/testData4.csv", index = False)
import pandas as pd
import random

df = pd.read_csv('C:/thesis-goettert/data/testData4.csv', sep = ',', quotechar = '"', dtype=str)

# ausgabe mit unique session ids
key = df['key']
key = list(dict.fromkeys(key))
number_of_keys = len(key)
print (f'Anzahl der keys: {number_of_keys}')


# random key ausgeben um bessere Stichproben auszugeben
#print(df[df['key'] == random.choice(key)])

# some test prints
#print(df[df['params'] == ""])
#print(df[df['params'].str.contains('OR OR')])
print(df[(df['mapping_action_label'].str.contains('searchterm', na=False)) & (df['params'].str.contains('fis-', na=False))])
#if df[df['params'] == 'action']:
#    print(df[df['params'].str.contains('fis-', na=False)])
#print(df[df['mapping_action_label'] == "searchterm_4"])
#print(df[df['key'] == "25eqk8ascc4g2r8o3ufpv8vvj5_0"])


# liste der keys deren mapping_action_label == goto_thesaurus in einer extra csv datei speichern
#keep_sessions = []
#for index, row in df.iterrows():
#    if (row['mapping_action_label'] == 'goto_thesaurus'):
 #       keep_sessions.append((row['key']))

#keep_sessions = list(dict.fromkeys(keep_sessions))
#df2 = pd.DataFrame(keep_sessions, columns=["key"])
#df2.to_csv('C:/thesis-goettert/data/thesaurus_IDS.csv', index=False)          
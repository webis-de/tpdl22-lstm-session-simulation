import pandas as pd
import random

df = pd.read_csv('C:/thesis-goettert/data/testData.csv', sep = ',', quotechar = '"', dtype=str)

# ausgabe mit unique session ids
unique_sessionIDs = df['session_id']
unique_sessionIDs = list(dict.fromkeys(unique_sessionIDs))
number_of_unique_ids = len(unique_sessionIDs)
print (f'Anzahl der einzigartigen IDs: {number_of_unique_ids}')

# ausgabe mit unique session ids + part ---> defnitiv neuer identifier für mein Problem
unique_sessionIDs_part = df['session_id'] + df['part'].astype(str)
unique_sessionIDs_part = list(dict.fromkeys(unique_sessionIDs_part))
number_of_unique_ids_part = len(unique_sessionIDs_part)
print (f'Anzahl der einzigartigen IDs + part: {number_of_unique_ids_part}')

# random session + part ausgeben um bessere Stichproben auszugeben
#print(df[df['session_id'] + df['part'] == random.choice(unique_sessionIDs_part)])
print(df[df['session_id'] + df['part'] == '0008csgt6g32ds76jncohjsl51j0'])
#print(df[df['session_id'] == 'sknpl5h8cv0ue20qs9hpld2ki2'])
#print(df[(df['part'].astype(int) >= 15)])
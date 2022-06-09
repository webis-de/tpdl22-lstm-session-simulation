import pandas as pd

df = pd.read_csv('C:/thesis-goettert/data/testData3.csv', sep = ',', quotechar = '"')

# liste zum speichern der keys
keep_sessions = []

# iterieren über jede row und speichern jedes keys bei dem ein query(searchterm) auftritt
for index, row in df.iterrows():
    if (row['mapping_action_label'] == 'searchterm_1') | (row['mapping_action_label'] == 'searchterm_2') | (row['mapping_action_label'] == 'searchterm_3') | (row['mapping_action_label'] == 'searchterm_4'):
        keep_sessions.append((row['key']))

# liste der keys in einer extra csv datei speichern
keep_sessions = list(dict.fromkeys(keep_sessions))
df2 = pd.DataFrame(keep_sessions, columns=["key"])
df2.to_csv('D:/thesis-goettert/data/keep_IDS.csv', index=False)          
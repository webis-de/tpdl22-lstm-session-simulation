import pandas as pd

df = pd.read_csv('C:/thesis-goettert/data/testData.csv', sep = ',', escapechar = '\\')

# zusammenfassen von sessio_id und part zum key
df['key'] = df['session_id'].astype(str) + '_' + df['part'].astype(str)

# nur behalten der relevanten Spalten
keep_col = ['key', 'log_id','part_length','date','action_length','part_step','mapping_type','mapping_action_label','params','origin_action']
new_file = df[keep_col]
new_file.to_csv("D:/thesis-goettert/data/testData2.csv", index = False)
df2 = pd.read_csv('D:/thesis-goettert/data/testData2.csv', sep = ',', quotechar = '"')
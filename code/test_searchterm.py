import pandas as pd

df = pd.read_csv('D:/thesis-goettert/data/testData3.csv', sep = ',', quotechar = '"')

#einzelnen Searchterms speichern um Unterschiede fest zu stellen
print(df[df['mapping_action_label'] == 'searchterm_1'])
df1 = df.drop(df[df.mapping_action_label != 'searchterm_1'].index)
df1['params'].sort_values(ascending=True).to_csv("D:/thesis-goettert/data/searchterm_1.csv", index = False)
print(df[df['mapping_action_label'] == 'searchterm_2'])
df2 = df.drop(df[df.mapping_action_label != 'searchterm_2'].index)
df2['params'].sort_values(ascending=True).to_csv("D:/thesis-goettert/data/searchterm_2.csv", index = False)
print(df[df['mapping_action_label'] == 'searchterm_3'])
df3 = df.drop(df[df.mapping_action_label != 'searchterm_3'].index)
df3['params'].sort_values(ascending=True).to_csv("D:/thesis-goettert/data/searchterm_3.csv", index = False)
print(df[df['mapping_action_label'] == 'searchterm_4'])
df4 = df.drop(df[df.mapping_action_label != 'searchterm_4'].index)
df4['params'].sort_values(ascending=True).to_csv("D:/thesis-goettert/data/searchterm_4.csv", index = False)
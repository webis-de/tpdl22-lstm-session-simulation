import pandas as pd

df = pd.read_csv('C:/thesis-goettert/data/testData2.csv', sep = ',', quotechar = '"')

delete_sessions = []

# alle negativen Werte in den Zeilen action_length und part_length durch 0 ersetzen
df.loc[df['action_length'] < 0, 'action_length'] = 0
df.loc[df['part_length'] < 0, 'part_length'] = 0

df.to_csv("C:/thesis-goettert/data/testData3.csv", index = False)

print(df[df['action_length'] < 0])
print(df[df['part_length'] < 0])
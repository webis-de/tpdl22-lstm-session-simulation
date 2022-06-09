import pandas as pd

df = pd.read_csv('C:/thesis-goettert/data/testData3.csv', sep = ',', quotechar = '"')
df2 = pd.read_csv('C:/thesis-goettert/data/keep_IDS.csv', sep = ',', quotechar = '"')
df = df.set_index(['key'])
df2 = df2.set_index(['key'])

#merge both keys together so only the keys with queries in it are being kept
df3 = pd.merge(df,df2,left_index=True, right_index=True)
df3.reset_index()
df3.to_csv("C:/thesis-goettert/data/testData4.csv", index = True) 
import pandas as pd
import os

df = pd.read_csv('C:/thesis-goettert/data/testData4.csv', sep = ',', quotechar = '"')

f = lambda x: x.to_csv(os.getcwd() + "/data_{}.csv".format(x.name), index=False)
df.groupby('key').apply(f)

for i, x in df.groupby('key'):
    p = os.path.join(os.getcwd(), "C:/thesis-goettert/data_/data_{}.csv".format(i))
    x.to_csv(p, index=False)
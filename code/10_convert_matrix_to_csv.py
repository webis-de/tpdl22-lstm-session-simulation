import pandas as pd
df = pd.read_json ('C:/thesis-goettert/Uebergangsmatrix/updated_matrix.json')
df.to_csv ('C:/thesis-goettert/Uebergangsmatrix/updated_matrix.csv', index = None)
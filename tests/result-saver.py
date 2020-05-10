import os.path
import pandas as pd

filePath = 'ml-results.csv'
if not os.path.isfile(filePath):
    f = open(filePath, "a")
    f.write("Name,Loss,Val_Loss,Mae,Val_Mae")
    f.close()

df = pd.read_csv(filePath)
df = df[['Name', 'Loss', 'Val_Loss', 'Mae', 'Val_Mae']]

df = df.append({'Name': 'Tom', 'Loss': 13.0, 'Val_Loss': 13.2, 'Mae': 12.4, 'Val_Mae': 9.0}, ignore_index=True)
print(df.head())

df.to_csv(path_or_buf=filePath, index=False)
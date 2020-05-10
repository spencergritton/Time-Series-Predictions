import pandas as pd

x_in_front = 5

df = pd.DataFrame({'close': [11,21,31,41,51,61,71,81,91,101]})
df["index"] = df.index.values.tolist()

def rowFunc(row):
    if row['index'] + 1 + x_in_front > len(row['Target']):
        return "na"
    return row['Target'][row['index'] + 1 : row['index'] + 1 + x_in_front]

df['Target'] = [df['close'].values.tolist()] * len(df['close'].values.tolist())
df['Target'] = df.apply(rowFunc, axis=1)
df = df[df.Target != "na"]
print(df.head(10))
import pandas as pd
import numpy as np

im= pd.read_csv('data.csv')
im.drop('Timestamp',axis=1,inplace=True)
def add_new_column(df):
  df['Play'] = ['Yes' for _ in df.index] 
  return df
im = add_new_column(im.copy())
checks =[]
for index,row in im.iterrows():
    val=[]
    for col in im.columns:
        val.append(row[col])
    checks.append(val)


def findNo(df,s=list()):
    columns  = df.columns.tolist()
    if(len(columns)==1):
        if (s+["Yes"] not in checks):
            checks.append(s+["No"])
        return
    for val in df[columns[0]].unique():
        findNo(df=df.drop(columns[0],axis=1),s=s+[val])
findNo(im)
im = pd.DataFrame(checks,columns=im.columns)
print(im)

paths = []

def Decision_Tree(data, target, path=[]):
    features = list(data.columns)
    if len(features) == 1:
        paths.append(path)
        return
    maxi = -1
    node = ""
    for val in features:
        if val == target:
            continue 
        val_entropy = 0 
        for value in data[val].unique():
            df = data[data[val] == value]
            ply_cnt = df[target].value_counts(normalize=True)
            df_entropy = -np.sum(ply_cnt * np.log2(ply_cnt))
            val_entropy += (len(df) / len(data)) * df_entropy
        gain = entropy - val_entropy
        if gain > maxi:
            maxi = gain
            node = val

    for i in data[node].unique():
        df = data[data[node] == i]
        df = pd.DataFrame(df, columns=[j for j in df if j != node])
        new_path = path + [{node: i}]
        d = df[target].value_counts(normalize=True)
        if len(d) == 1:
            if d.index.tolist()[0]=='Yes':
                new_path+=[{'res':d.index.tolist()[0]}]
                paths.append(new_path)
            continue
        else:
            Decision_Tree(df, target, new_path)


target = 'Play'
play_counts = im[target].value_counts(normalize=True)
entropy = -np.sum(play_counts * np.log2(play_counts))
Decision_Tree(im, target)
print("conditions for playing: ")
for i in paths:
    print (i)

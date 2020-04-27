'''
Compute general statistics
'''

import pandas as pd

def load_dataset(path):
    return pd.read_csv(path)

def PEARSONR(data):
    columns=data.columns
    values = []
    values.append(list(round(data.min(),2)))
    values.append(list(round(data.max(),2)))
    values.append(list(round(data.std(),2)))
    values.append(list(round(data.mean(),2)))
    values.append(list(round(data.median(),2)))
    values.append(list(round(data.mode(axis=1),2)))
    values.append(list(round(data.quantile(0.25),2)))
    values.append(list(round(data.quantile(0.75),2)))
    values = pd.DataFrame(values,columns=columns,index=["MIN","MAX","STDEV","MEAN","MEDIAN","MODE","1stQuantile","3rdQuantile"])
    values.to_csv("CSV_Files/DescriptiveAnalysis.csv")
    print(values)

def main():
    path = "CSV_Files/outliers_removed.csv"
    data = load_dataset(path)
    data.drop(columns=["CreationTime"],inplace=True)
    PEARSONR(data)


if __name__=='__main__':
    main()
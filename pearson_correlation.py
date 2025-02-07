'''
PearsonR
'''

from scipy.stats import pearsonr
import pandas as pd

def load_dataset(path):
    return pd.read_csv(path)

def PEARSONR(data):
    values = []
    columns = data.columns
    for i in range(len(columns)):
        l = []
        for j in range(len(columns)):
            l.append(round(pearsonr(data[columns[i]],data[columns[j]])[0],2))
        values.append(l)
    values = pd.DataFrame(values,columns=columns,index=columns)
    values.to_csv("CSV_Files/PearsonCoefficientValues.csv")
    print(values)

def main():
    path = "CSV_Files/outliers_removed.csv"
    data = load_dataset(path)
    data.drop(columns=["CreationTime"],inplace=True)
    PEARSONR(data)


if __name__=='__main__':
    main()

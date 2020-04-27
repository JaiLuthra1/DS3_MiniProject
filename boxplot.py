'''
Boxplots
'''

import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('Problem/group7.csv')
df = df.drop(df.columns[0], axis=1)
for column in df.columns:
    plt.xlabel(column)
    plt.boxplot(df[column])
    plt.savefig('Plots/{}.jpg'.format('column'))
    plt.show()
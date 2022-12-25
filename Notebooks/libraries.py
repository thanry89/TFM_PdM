import pandas as pd
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt


def timeSeq(df, tolerance):
    dif = pd.DataFrame(df.time.diff(periods=1))
    empty = [np.nan for i in range(24)]
    seq = []
    data = []
    for i in df.index:
        if dif.time[i] > tolerance:
            for j in range(df.time[i-1]+4, df.time[i], 4):
                seq.append([j,1])
                data.append(empty)
            seq.append([df.time[i], 0])
            data.append(df.iloc[i].tolist())
        else:
            seq.append([df.time[i], 0])
            data.append(df.iloc[i].tolist())
    fill = pd.DataFrame(seq, columns =['time', 'filled'])
    data = pd.DataFrame(data, columns = df.columns)
    data[['time','filled']] = fill[['time','filled']]
    return data



                    
def refill(dfOrigin, dfDestiny, start, end):
    for column in dfOrigin.columns:
        dfRange = dfOrigin[column].iloc[start:end]
        it = cycle(dfRange)
        for x in dfDestiny[dfDestiny[column].isnull()].index:
            dfDestiny.at[x, column] = next(it)
    return dfDestiny
        


    
def faultData(data, dataFault):
    dataset = data.copy()
    dataset['fault_type'] = 'No Fault'
    dataset['Target'] = 'No Fault'
    for i in dataFault.index:
        idx = dataset[dataset['time']>=dataFault['time'][i]].iloc[0,:].name
        dataset.at[idx,'fault_type'] = dataFault['fault_name'][i]
        dataset.at[idx,'Target'] = 'Fault'    
    return dataset



def dibujo(df, column):
    fig, ax = plt.subplots(1, 1, figsize=(12, 4), constrained_layout=True)
    line, = ax.plot('time', column, 'o', markersize=0.5, data=df, lw=2.5)
    ax.set_title(column, fontsize='small', loc='left')
    ax.grid()
    fig.supxlabel('time')
    plt.show()
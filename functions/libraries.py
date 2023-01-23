# Importación de Funciones
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay



# Función para completar secuencia de datos
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




# Función para incluir datos de falla al dataset    
def faultData(data, dataFault):
    dataset = data.copy()
    dataset['fault_type'] = 'No Fault'
    dataset['fault'] = 'No Fault'
    for i in dataFault.index:
        idx = dataset[dataset['time']>=dataFault['time'][i]].iloc[0,:].name
        dataset.at[idx,'fault_type'] = dataFault['fault_name'][i]
        dataset.at[idx,'fault'] = 'Fault'    
    return dataset



   
# Función para aplicar normalización estándar 
def scaling(data):
    scaler = StandardScaler()
    X = scaler.fit_transform(data)
    return X




# Función para aplicar codificación
def encoding(data):
    leFault = LabelEncoder()
    fault = leFault.fit_transform(data['fault'])
    for i in np.unique(fault):
        print(i, leFault.inverse_transform([i]))
    return fault




# Función para presentar los resultados de la predicción
def predicted (time, real, predicted, faultType, window):
    list_of_tuples = list(zip(time.tolist(), real.tolist(), 
                          predicted.tolist(), faultType.tolist()))
    df = pd.DataFrame(list_of_tuples, columns=['time', 'Real', 'Predicted', 'Type'])
    df['predTime'] = np.nan
    df['false'] = np.nan
    print ("{:<10} {:<10} {:<10} {:<10}".format('timestamp','Predicted','hours', 'Type'))
    predCount = 0
    noPredCount = 0
    types = []
    #idxFault = []
    finalPred = pd.DataFrame(1, index=predicted.index, columns=['predicted'])
    for i in np.where(df.Real == 0)[0]:
        datos = np.where(df['Predicted'][0:i] == 0)
        if len(datos[0]) == 0:
            x = window
        else:
            tiempo = [df.iloc[x]['time'] for x in datos[0]]
            dif = pd.DataFrame(list(zip(datos[0].tolist(), tiempo)), 
                               columns = ['idx','time'])
            dif['diference'] = df.iloc[i]['time'] - dif['time']          
            x = dif.iloc[-1].diference
            for index, row in dif[dif.diference < window].iterrows():
                df.at[row.idx,'predTime'] = row.diference
                df.at[row.idx,'false'] = 1
            for index, row in dif[dif.diference >= window].iterrows():
                if df['false'].iloc[row.idx] != 1:
                    finalPred.at[row.idx,'predicted'] = 0
        if x < window:
            pred = (x/3600).round(2)
            print ("{:<10} {:<10} {:<10} {:<10}".format(df.time[i], 'YES', pred, df.Type[i]))
            predCount = predCount + 1
            finalPred.at[i,'predicted'] = 0
            types.append(df.Type[i])
        else:
            print ("{:<10} {:<10} {:<10} {:<10}".format(df.time[i],'NO',np.nan, np.nan))
            noPredCount = noPredCount + 1
    print("El total de fallas reales son: %d" % real.value_counts()[0])
    print("El total de fallas predecidas por el modelo son: %d" % predicted.value_counts()[0])
    print("Las fallas predecidas fueron: %d" % predCount)
    print("Las fallas NO predecidas fueron: %d" % noPredCount)
    print("EL conteo de tipos de falla es:")
    print(pd.DataFrame(types).value_counts())
    print("La precisión del modelo es del %.2f" % (predCount/real.value_counts()[0]))
    #for i in idxFault:
    #    finalPred.at[i, 'predicted'] = 0
    return finalPred




def modelMetrics(real, predicted):
    cm = confusion_matrix(real, predicted)
    score = f1_score(real, predicted)
    disp = ConfusionMatrixDisplay(cm, display_labels= ['Fault', 'No Fault'])
    disp.plot()
    print("El f1-score del Modelo es: %.20f" % score)
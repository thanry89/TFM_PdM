# Importar librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import time
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pyod.models.auto_encoder import AutoEncoder




def KNNModel(data, faultData):
    start = time()
    print("Entrenamiento del modelo")
    nbrs = NearestNeighbors(n_neighbors=2)
    nbrs.fit(data)
    print("El tiempo transcurrido es: %2f [segundos]" % (time()-start))
    print("Predicción del modelo")
    distances, indexes = nbrs.kneighbors(data)
    distancias = pd.DataFrame(distances)
    distanciasProm = distancias.mean(axis=1)
    realFaults = faultData.value_counts()[0]
    predFaults = pd.Series(0)
    percent=0.5
    while predFaults.value_counts()[0] < realFaults:
        th=np.max(distanciasProm)*percent
        outlierIdx = np.where(distanciasProm > th)
        predFaults = pd.Series(1, index = distanciasProm.index)
        predFaults.iloc[outlierIdx] = 0
        predFaults.value_counts()
        percent = percent - 0.01
    print("El tiempo transcurrido es: %2f [minutos]" % ((time()-start)/60))
    print("Resultados:") 
    print("Real")
    unique, counts = np.unique(faultData, return_counts=True)
    print(np.asarray((unique, counts)).T) 
    print("Predicted")
    unique, counts = np.unique(predFaults, return_counts=True)
    print(np.asarray((unique, counts)).T)
    print("Plot de Resultados")
    pred = pd.DataFrame(predFaults, index = faultData.index)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(faultData, label = 'Real', lw = 0.5)
    ax.plot(pred, color = 'r',  label = 'Predicted', lw = 0.5)
    ax.grid()
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2)
    print("La duración del modelo ha sido: %2f [minutos]" % ((time()-start)/60))
    return pred




def LOFModel (data, target):
    start = time()
    print("Entrenamiento del modelo")
    ctm = target[target == 0].shape[0] / data.shape[0]
    clf = LocalOutlierFactor(n_neighbors=3, contamination = ctm)
    print("El tiempo transcurrido es: %2f [segundos]" % (time()-start))
    print("Predicción del modelo")
    pred = clf.fit_predict(data)
    pred = [0 if i==-1 else 1 for i in pred]
    print("El tiempo transcurrido es: %2f [minutos]" % ((time()-start)/60))
    print("Resultados:") 
    print("Real")
    unique, counts = np.unique(target, return_counts=True)
    print(np.asarray((unique, counts)).T) 
    print("Predicted")
    unique, counts = np.unique(pred, return_counts=True)
    print(np.asarray((unique, counts)).T)
    print("Plot de Resultados")
    pred = pd.DataFrame(pred, index = target.index)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(target, label = 'Real', lw = 0.5)
    ax.plot(pred, color = 'r',  label = 'Predicted', lw = 0.5)
    ax.grid()
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2)
    print("La duración del modelo ha sido: %2f [minutos]" % ((time()-start)/60))
    return [pred, ctm]




def OneClassSVMModel (data, target):
    start = time()
    print("Entrenamiento del modelo")
    ctm = target[target == 0].shape[0] / data.shape[0]
    model = OneClassSVM(nu=ctm, kernel = 'sigmoid', gamma = 'auto')
    model.fit(data)
    print("El tiempo transcurrido es: %2f [minutos]" % ((time()-start)/60))
    print("Predicción del modelo")
    pred = model.predict(data)
    pred = [0 if i==-1 else 1 for i in pred]
    print("El tiempo transcurrido es: %2f [minutos]" % ((time()-start)/60))
    print("Resultados:") 
    print("Real")
    unique, counts = np.unique(target, return_counts=True)
    print(np.asarray((unique, counts)).T)
    print("Predicted")
    unique, counts = np.unique(pred, return_counts=True)
    print(np.asarray((unique, counts)).T)
    print("Plot de Resultados")
    pred = pd.DataFrame(pred, index = target.index)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(target, label = 'Real', lw = 0.5)
    ax.plot(pred, color = 'r',  label = 'Predicted', lw = 0.5)
    ax.grid()
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2)
    print("La duración del modelo ha sido: %2f [minutos]" % ((time()-start)/60))
    return [pred, ctm]





def AutoEncoderModel(data, target, rep):
    start = time()
    print("Entrenamiento del modelo")
    ctm = 0.015
    atcdr = AutoEncoder(contamination=ctm, hidden_neurons =[2, 2], epochs = rep)
    atcdr.fit(data)
    print("El tiempo transcurrido es: %2f [minutos]" % ((time()-start)/60))
    print("Predicción del modelo")
    y_test_pred = atcdr.predict(data)
    pred = 1 - y_test_pred
    print("El tiempo transcurrido es: %2f [minutos]" % ((time()-start)/60))
    print("Resultados:") 
    print("Real")
    unique, counts = np.unique(target, return_counts=True)
    print(np.asarray((unique, counts)).T)
    print("Predicted")
    unique, counts = np.unique(pred, return_counts=True)
    print(np.asarray((unique, counts)).T)
    print("Plot de Resultados")
    pred = pd.DataFrame(pred, index = target.index)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(target, label = 'Real', lw = 0.5)
    ax.plot(pred, color = 'r',  label = 'Predicted', lw = 0.5)
    ax.grid()
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2)
    print("La duración del modelo ha sido: %2f [minutos]" % ((time()-start)/60))
    return [pred, ctm]
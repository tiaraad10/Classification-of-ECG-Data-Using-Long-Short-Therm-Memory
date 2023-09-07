# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:02:18 2020

@author: Jannes :3
"""
# dibaca komenny biar dk salah.
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.layers import Dense,LSTM, Bidirectional, Dropout,Conv1D, MaxPool1D, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.models import load_model

import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.model_selection import train_test_split
plt.ioff()

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


"""
Model = Model yang sudah di train,
nama = nama label y
jenis plot = akurasi or loss ?
labels = nama label
"""
def plot_akurasi_loss(modell,nama,jenis_plot=['acc','val_acc'],labels=['Akurasi Training','Akurasi Testing']):
    fig,(ax0) = plt.subplots(nrows=1, figsize=(13,8))
    ax0.plot(modell.history.history[jenis_plot[0]],'green', label=labels[0],linewidth=4)
    ax0.plot(modell.history.history[jenis_plot[1]],'red', label=labels[1],linewidth=4)
    ax0.tick_params(axis="x", labelsize=30)
    ax0.tick_params(axis="y", labelsize=30)

    ax0.set_title(nama[0], size=40)
    ax0.set_xlabel("Epoh", size=30)
    ax0.set_ylabel(nama[1], size=30)
    ax0.legend(borderaxespad=0., prop={'size': 30})

# Fungsi ROC
"""
Data_gt = label ground truth
data_prediksi = label prediksi
nama model = nama model untuk forma nama save gambar.

format data 2 dimensi (semua_titik,Jumlah kelas)
misall data(200,2500), label(200,7) -> masukkan (200*2500,7)

contoh penggunaan data delineasi ( 3 dimensi)
nama_model = "cnn 8 kelas"
ukuran = test_label.shape[0]*test_label.shape[1] 
n_kelas= test_label.shape[2]

testing_predicted = model.predict(test_data)
testing_rounded = testing_predicted.round()
testing_rounded = testing_rounded.reshape((ukuran,n_kelas))
gt_testing = test_label.reshape((ukuran,n_kelas))
ROC_PR(gt_testing,testing_rounded,nama_model)
"""
def ROC_PR(data_gt,data_prediksi,nama_model):
  import numpy as np
  import matplotlib.pyplot as plt
  from scipy import interp
  from sklearn.metrics import roc_curve, auc
  from itertools import cycle
  from sklearn.metrics import precision_recall_curve
  from sklearn.metrics import average_precision_score
  import glob

  list_test_label = data_gt.copy()
  list_test_predict = data_prediksi.copy()

  model = nama_model
  n_classes = 7

  #Inisialisasi 
  fpr = dict()
  tpr = dict()
  roc_auc = dict()
  lw = 2
  for i in np.arange(n_classes):
      fpr[i], tpr[i], _ = roc_curve(list_test_label[:, i], list_test_predict[:, i])
      roc_auc[i] = auc(fpr[i], tpr[i])
      
  # First aggregate all false positive rates
  all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

  # Then interpolate all ROC curves at this points
  mean_tpr = np.zeros_like(all_fpr)
  for i in np.arange(3):
      mean_tpr += interp(all_fpr, fpr[i], tpr[i])

  # Finally average it and compute AUC

  mean_tpr /= n_classes

  fpr["macro"] = all_fpr
  tpr["macro"] = mean_tpr
  roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

  # Plot all ROC curves
  plt.figure(figsize=(15,10))

  #colors = cycle(['aqua', 'darkorange', 'black','red','green','pink','purple','yellow','brown','grey'])
  colors = cycle(['aqua', 'darkorange', 'black','red','green','pink','purple'])
  kelas = ['Gelombang P','Poff-Qon','Qon-Rpeak','Rpeak-Qoff','Qoff-Ton','Gelombang T','Toff-Pon','Bukan Gelombang']
  #colors = cycle(['aqua', 'darkorange', 'black'])
  for i, color in zip(np.arange(n_classes), colors):
      plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='Kurva ROC Kelas {0} (area = {1:0.2f})'.format(kelas[i], roc_auc[i]))

  plt.plot([0, 1], [0, 1], 'k--', lw=lw)
  plt.xlim([0, 1])
  plt.ylim([0.00001, 1.05])
  plt.xlabel('False Positive Rate',size=20)
  plt.ylabel('True Positive Rate',size=20)
  plt.title('Kurva ROC 7 Kelas Klasifikasi Sinyal Elektrokardiogram',size=25)
  plt.tick_params(axis="x", labelsize=30)
  plt.tick_params(axis="y", labelsize=30)

  plt.legend(borderaxespad=0., prop={'size':20}) # Prop SIze adalah Ukuran Label

  plt.savefig('ROC curve fix.jpg',dpi=250,bbox_inches='tight')
  plt.close()

  # =============================================================================
  # PR PLOT
  # =============================================================================
  
  # For each class
  precision = dict()
  recall = dict()
  average_precision = dict()
  for i in np.arange(7):
      precision[i], recall[i], _ = precision_recall_curve(list_test_label[:, i],
                                                          list_test_predict[:, i])
      average_precision[i] = average_precision_score(list_test_label[:, i], list_test_predict[:, i])

      
  # setup plot details
  colors = cycle(['aqua', 'darkorange', 'black','red','green','pink','purple'])

  plt.figure(figsize=(15, 10))
  f_scores = np.linspace(0.2, 0.8, num=4)
  lines = []
  labels = []
  for f_score in f_scores:
      x = np.linspace(0.01, 1)
      y = f_score * x / (2 * x - f_score)
      l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
      plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02),size=25)

  lines.append(l)
  labels.append('iso-f1 curves')
  #l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
  #lines.append(l)
  #labels.append('micro-average Precision-recall (area = {0:0.2f})'
          #      ''.format(average_precision["micro"]))

  for i, color in zip(np.arange(n_classes), colors):
      l, = plt.plot(recall[i], precision[i], color=color, lw=2)
      lines.append(l)
      labels.append('Presisi-Recall Kelas {0} (area = {1:0.2f})'.format(kelas[i], average_precision[i]))
  fig = plt.gcf()
  # fig.subplots_adjust(bottom=0.25)
  plt.xlim([0.0, 1.0])
  plt.ylim([0.00001, 1.05])
  plt.tick_params(axis="x", labelsize=23)
  plt.tick_params(axis="y", labelsize=23)
  plt.xlabel('Recall',size=20)
  plt.ylabel('Presisi',size=20)
  plt.title('Kurva Presisi-Recall 7 Kelas Klasifikasi Sinyal Elektrokardiogram',size=25)
  plt.legend(lines, labels,fontsize=28, loc="lower left", prop=dict(size=20)) 
  # loc = lokasi label
  # tick param , Memperbesar atau memperkecil jarak pandang

  plt.savefig('PR curve cba fix.jpg',dpi=250,bbox_inches='tight')
  plt.close()
  return 0


data_gt = np.loadtxt('cnn lstm/roc/ground truth testing 11-6-2021--CNN-LSTM-8-kelas- 1 Layer LR 0.01 Batch Size 8.csv',delimiter=',')
prediksi = np.loadtxt('cnn lstm/roc/prediksi testing 11-6-2021--CNN-LSTM-8-kelas- 1 Layer LR 0.01 Batch Size 8.csv',delimiter=',')
nama_model = '7 Kelas Klasifikasi Sinyal Elektrokardiogram'

ROC_PR(data_gt,prediksi,nama_model)
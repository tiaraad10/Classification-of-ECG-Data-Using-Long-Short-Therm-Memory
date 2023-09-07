# -*- coding: utf-8 -*-

# dibaca komenny biar dk salah.
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import tensorflow as tf
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
# tf.config.experimental.set_memory_growth(physical_devices[1], enable=True)
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.layers import Dense,LSTM, Dropout, MaxPool1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.optimizers import Adam,SGD

import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.model_selection import train_test_split


"""
Membuat fungsi model untuk membuat banyak model
"""
def make_model(ukuran_input,learning_rate,output,n_layer=1):
    opt = Adam(lr=learning_rate)
    modell = Sequential()
    modell.add(LSTM(512,input_shape=ukuran_input,return_sequences=True))
    for i in range(n_layer-1):
        model.add(LSTM(512,input_shape=ukuran_input,return_sequences=True))
    # modell.add(Dropout(0.2))
    modell.add(Dense(output,activation='softmax'))
    modell.compile(optimizer=opt,loss="categorical_crossentropy",metrics=['acc'])
    return modell

def load_file(nama_file):
    with open(nama_file, 'rb') as fread:
        data = pickle.load(fread)
    return data

def plot_akurasi_loss(modell,nama,jenis_plot=['acc','val_acc'],labels=['Akurasi Training','Akurasi Testing']):
    fig,(ax0) = plt.subplots(nrows=1, figsize=(13,8))
    ax0.plot(modell.history.history[jenis_plot[0]],'blue', label=labels[0],linewidth=4)
    ax0.plot(modell.history.history[jenis_plot[1]],'red', label=labels[1],linewidth=4)
    ax0.tick_params(axis="x", labelsize=30)
    ax0.tick_params(axis="y", labelsize=30)

    ax0.set_title(nama[0], size=40)
    ax0.set_xlabel("Epoch", size=30)
    ax0.set_ylabel(nama[1], size=30)
    ax0.legend(borderaxespad=0., prop={'size': 30})


train_data  = load_file('pickles2/train data 8 kelas')
train_label = load_file('pickles2/train label 8 kelas')
test_data  = load_file('pickles2/test data 8 kelas')
test_label = load_file('pickles2/test label 8 kelas')


# all_sinyal_rnn = load_file('data pickle/Beat Based/semua normal qtdb 4 kelas')
# all_labels_rnn = load_file('data pickle/Beat Based/semua label normal qtdb 4 kelas')
# train_data,test_data,train_label, test_label = train_test_split(all_sinyal_rnn,all_labels_rnn,test_size=0.1,random_state=42)


all_list_sen = []
all_list_spe = []
all_list_pre = []
all_list_f1 = []
all_list_err = []
all_list_acc = []


# Mulai Tuning
tanggal ='20-4-2021'
x_kelas = 8
lr = [0.001,0.001,0.0001,0.00001]
n_layer = [1,2,3,4,5]
batch_size = [8,16,32,64]
nama = "LUDB 8 Kelas"

panjang = test_label.shape[1]
"""
MANUAL HYPERTUNING PARAMETER UNTUK LEARNING RATE DAN BATCH SIZE
"""

# TRAINING DAN TUNING BATCH SIZE DARI 8 - 64, KELIPATAN 8
#jangan lupa ubah nama_model untuk tiap tuning 
#jangan lupa ubah batch dan lr tiap tuning
# i = jumlah layer
# j = LR
# k = batch_size
i = 0
j = 1
k = 1
for i in range(len(n_layer)):
        nama_model = '{}--LSTM-8-kelas- {} Layer LR {} Batch Size {}'.format(tanggal,n_layer[i],lr[j],batch_size[k])
        
        path_model = 'lstm/tuning/model/'+nama_model+'.h5'
        path_model_best = 'lstm/tuning/best model/best '+nama_model+'.h5'
        path_plot_acc = 'lstm/tuning/plot/plot akurasi '+nama_model+'.jpg'
        path_plot_loss = 'lstm/tuning/plot/plot loss '+nama_model+'.jpg'
        path_waktu = 'lstm/tuning/time/training duration'+nama_model+'.txt'
        path_acc = 'lstm/tuning/time/akurasi model '+nama_model+'.txt'
        
        #save best model
        mc = ModelCheckpoint(path_model_best,monitor='val_acc',mode='max', verbose=1, save_best_only=True)
        # es = EarlyStopping(monitor='val_acc', mode='max', verbose=1)
        
        start_time = time.time()
        ukuran_data = (train_data.shape[1],train_data.shape[2])
        output = train_label.shape[2]
        
        # JANGAN LUPO UBAH nlayer,batch,lr di sini
        model = make_model(ukuran_data,learning_rate =lr[j],output=output,n_layer = n_layer[i])
        model.summary()
        with tf.device("/device:GPU:0"):
            model.fit(train_data,train_label,epochs=300,batch_size=batch_size[k],validation_data=(test_data,test_label),callbacks=[mc])
        lama = time.time() - start_time
        
        #save model
        model.save(path_model)
        
        lama = lama / 3600
        #lama Training
        with open(path_waktu, 'w') as fsave:
            fsave.write(str(lama))
        
        #lama Training
        akurasis = model.history.history['val_acc'][-1]
        with open(path_acc, 'w') as fsave:
            fsave.write(str(akurasis))
        
        acc_history = model.history.history['acc']
        acc_history = np.array(acc_history)
        np.savetxt('lstm/acc history/acc history'+nama_model+'.csv', acc_history, delimiter=',', fmt='%f')
    
        val_acc_history = model.history.history['val_acc']
        val_acc_history = np.array(val_acc_history)
        np.savetxt('lstm/acc history/val_acc_history '+nama_model+'.csv', val_acc_history, delimiter=',', fmt='%f')
    
        loss_history = model.history.history['loss']
        loss_history = np.array(loss_history)
        np.savetxt('lstm/acc history/loss_history'+nama_model+'.csv', loss_history, delimiter=',', fmt='%f')
    
        val_loss_history = model.history.history['val_loss']
        val_loss_history = np.array(val_loss_history)
        np.savetxt('lstm/acc history/val_loss_history'+nama_model+'.csv', val_loss_history, delimiter=',', fmt='%f')
        
        #Plotting Akurasi
        plot_akurasi_loss(model,nama=["Akurasi Model","Akurasi"])
        plt.savefig(path_plot_acc,dpi=250,bbox_inches='tight')
        plt.close()
        
        #Plotting Loss
        plot_akurasi_loss(model,nama=["Loss Model","Loss"],jenis_plot=['loss','val_loss'],labels=['Loss Training','Loss Testing'])
        plt.savefig(path_plot_loss,dpi=250,bbox_inches='tight')
        plt.close()
        
        # Mulai Hitung CM
        ukuran = test_data.shape[0]*test_data.shape[1]
        n_kelas= test_label.shape[2]
        
        with tf.device("/GPU:0"):
            testing_predicted = model.predict(test_data)
        testing_rounded = testing_predicted.round()
        testing_rounded = testing_rounded.reshape((ukuran,n_kelas))
        
        gt_testing = test_label.reshape((ukuran,n_kelas))
        y_predict =testing_rounded.argmax(axis=1)
        y_asli =gt_testing.argmax(axis=1)
        
    # =============================================================================
    #               save ke csv untuk ROC dan PR Curve
    # =============================================================================
    
        path_roc = 'lstm/roc/prediksi testing '+nama_model+'.csv'
        path_roc_gt = 'lstm/roc/ground truth testing '+nama_model+'.csv'
        
        np.savetxt(path_roc,testing_rounded , delimiter=',', fmt='%f')
        np.savetxt(path_roc_gt,gt_testing , delimiter=',', fmt='%f')
        
    # =============================================================================
    #     Untuk Plot beat by beat
    # =============================================================================
        size = test_data.shape[0]
        
        path_gt_sinyal = 'lstm/ground truth/{} '.format(nama)+nama_model+'.csv'
        path_gt_prediksi = 'lstm/ground truth/prediksi {} '.format(nama)+nama_model+'.csv'
        path_gt = 'lstm/ground truth/GT {} '.format(nama)+nama_model+'.csv'
    
        np.savetxt(path_gt_sinyal,X= test_data.reshape(size,panjang).T,delimiter =",")
        np.savetxt(path_gt,X= y_asli.reshape(size,panjang).T,fmt='%d',delimiter =",")
        np.savetxt(path_gt_prediksi,X= y_predict.reshape(size,panjang).T,fmt='%d',delimiter =",")
        
    
        index_cm = ["PWave","Poff-Qon","Qon-Rpeak","Rpeak-Qoff","Qoff-Ton","Twave","Toff-Pon","Zero Pad"]
        cm_testing = confusion_matrix(y_asli,y_predict)
        path_cm = 'lstm/confusion matrix/CM {} '.format(nama)+nama_model+'.csv'
        
        pd_cm_testing = pd.DataFrame(cm_testing,columns=index_cm[:x_kelas],index = index_cm[:x_kelas])
        pd_cm_testing.to_csv(path_cm)
        
        Pengukuran =[]
        Spe_Class = []
        Pre_Class = []
        F1_Class = []
        Err_Class = []
        Sen_Class = []
        Acc_Class = []
        list_sen = []
        list_spe = []
        list_pre = []
        list_f1 = []
        list_err = []
        list_acc = []
        
        for idx in range(len(cm_testing)):
            TP = cm_testing[idx, idx]
            FN = np.sum(cm_testing[idx, :]) - TP
            FP = np.sum(cm_testing[:, idx]) - TP
            TN = cm_testing.trace() - TP
            
            Recall = TP / (TP + FN)
            Presisi = TP / (TP + FP)
            Spesifity = TN / (TN + FP)
            Akurasi = (TP+TN) / (TP+FP+FN+TN)
            Error = (FP + FN) / (FP + FN + TN + TP)
            F1 = (2 * Presisi * Recall) / (Recall + Presisi)
            Pengukuran.append([Recall,Presisi,Spesifity,Akurasi,Error,F1])
            
            Sen_Class.append([Recall, idx])
            Spe_Class.append([Spesifity, idx])
            Pre_Class.append([Presisi, idx])
            F1_Class.append([F1, idx])
            Err_Class.append([Error, idx])
            Acc_Class.append([Akurasi, idx])
            
        list_sen.extend(Sen_Class)
        list_spe.extend(Spe_Class)
        list_pre.extend(Pre_Class)
        list_f1.extend(F1_Class)
        list_err.extend(Err_Class)
        list_acc.extend(Acc_Class)
        
        save_list_sen = np.array(list_sen)
        save_list_spe = np.array(list_spe)
        save_list_pre = np.array(list_pre)
        save_list_f1 = np.array(list_f1)
        save_list_err = np.array(list_err)
        save_list_acc = np.array(list_acc)
        
        save_average_sen = np.mean(save_list_sen[:,0]).reshape(-1,1)
        save_average_spe = np.mean(save_list_spe[:,0]).reshape(-1,1)
        save_average_pre = np.mean(save_list_pre[:,0]).reshape(-1,1)
        save_average_f1 = np.mean(save_list_f1[:,0]).reshape(-1,1)
        save_average_err = np.mean(save_list_err[:,0]).reshape(-1,1)
        save_average_acc = np.mean(save_list_acc[:,0]).reshape(-1,1)
        
        all_list_sen.append(save_average_sen[0])
        all_list_spe.append(save_average_spe[0])
        all_list_pre.append(save_average_pre[0])
        all_list_f1.append(save_average_f1[0])
        all_list_err.append(save_average_err[0])
        all_list_acc.append(save_average_acc[0])
            
        Pengukuran_CM = ["recall","presisi","Spesifity","Akurasi","Error","F1"]
        CM_semuaKelas = {"Pengukuran cm" : Pengukuran_CM,
                       "PWave" : Pengukuran[0],
                       "Poff-Qon" :Pengukuran[1],
                       "Qon-Rpeak" :Pengukuran[2],
                       "Rpeak-Qoff" :Pengukuran[3],
                       "Qoff-Ton" :Pengukuran[4],
                       "Twave" :Pengukuran[5],
                       "Toff-Pon" :Pengukuran[6],
                       "Zero Pad" :Pengukuran[7]}
        df_CM_semuaKelas = pd.DataFrame(CM_semuaKelas)
        df_CM_semuaKelas.to_csv('lstm/confusion matrix/Pengukuran CM {} '.format(nama)+nama_model+'.csv')
    
    
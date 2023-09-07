# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import glob
import wfdb
from wfdb.processing import normalize_bound
from wavelet import wavelet
from sklearn.model_selection import train_test_split

"""
1. baca data
2. ambil informasi yang diperlukan
3. preprocessing Sinyal
4. labelling sinyal
5. padding zero
6. split train test
"""



def save_file(nama_file,data):
    import pickle
    with open(nama_file, 'wb') as fsave:
        pickle.dump(data, fsave)

# Menlist semua path data sinya .dat
file_dat = glob.glob('D:/sinyal/tiara/ludb/*.dat')

# Men Split semua path data tanpa .dat
all_file = []
for i in range(len(file_dat)):
    file = file_dat[i].split(".")[0]
    all_file.append(file)


nilai_max = 0
all_sinyal = []
all_label = []
all_sinyal_ukuran_sama= []
all_label_ukuran_sama=[]
threshold_max = 400
minx = 1000
n_kelas = 5

"""
Membaca semua file sinyal
"""
for i in range(len(all_file)):
    annotation = wfdb.rdann(all_file[i], 'atr_ii', sampfrom=0,sampto=None)
    ann_dict = annotation.__dict__
    
    sample = ann_dict["sample"]
    symbol = ann_dict["symbol"]
    
    record = wfdb.rdrecord(all_file[i],sampfrom=0,sampto=None)
    record_dict = record.__dict__
    
    #mengambil data raw sinyal.
    sinyal_lead2 = record_dict["p_signal"][:,1]
    
    # Melakukan proses dwt terhadap sinyal yang sudah di normalisasi dengan level 8 dan bior6.8
    dwt_sinyal = wavelet(sinyal_lead2,8,family="bior6.8")
    
    # Normalisasi data sinyal ecg 1 dengan nilai 0 - 1
    clean_sinyal= normalize_bound(dwt_sinyal)
    tanda = 0
    for i in range(len(symbol)):
        """
        Pengecekan apakah satu beat terdiri atas P QRS T yang normal,
            Kondisi tanda dengan anggapan 
            P = 1
            QRS = 2
            T = 3
            kalau bukan bearti pasti bukan satu beat
        """
        if symbol[i]=="p":
            tanda = tanda + 1

            if tanda == 1:
                Pon = sample[i-1]
                Poff = sample[i+1]
                
                SPon = symbol[i-1]
                SPoff = symbol[i+1]
                if SPon != '(' or SPoff !=')':
                    tanda =0
                    continue
            else: 
                tanda = 0
                
        elif symbol[i]=="N":
            tanda = tanda + 1
            if tanda == 2:
                Qon = sample[i-1]
                Qoff = sample[i+1]
                SQon = symbol[i-1]
                SQoff = symbol[i+1]
                Rpeak = sample[i]
                if SQon != '(' or SQoff !=')':
                    tanda =0
                    continue
            else:
                tanda = 0    
                
        elif symbol[i]=="t":
            if symbol[i+1]=="t": # kondisi jika ( t  t )
                tanda = 0
                continue
            tanda = tanda + 1
            if tanda ==3:
                #kondisi kalo t on dan t off tidak ada, -->  ... t )
                if symbol[i-1] != '(' or symbol[i+1] != ')':
                    tanda = 0
                    continue
                
                Ton = sample[i-1]
                Toff= sample[i+1]
                try:
                    # untuk kondisi p,qrs,t,qrs,t
                    if symbol[i+3]=='p':
                        Pon2 = sample[i+2]
                    else:
                        tanda = 0
                        continue
                except IndexError:
                    break
                
                """
                memotong sinyal 1 beat yang terdiri atas P QRS T yang normal 
                """
                Sinyal_satu_beat = clean_sinyal[Pon:Pon2]

                
                # Mencari Nilai Max dari semua beat, supaya bisa di padding
                max_sementara = len(Sinyal_satu_beat) 
                if max_sementara > nilai_max:
                    nilai_max = max_sementara
                    
              
                """
                Konsep Labelling
                1. Buat Label Kosong dengan ukuran 1 beat.
                2. Label 6 kelas dari Pon ke Toff
                3. Transpose untuk one hot encoding
                """
                label = np.zeros((8,Pon2-Pon),dtype="int")
                Anotasi = list([Pon,Poff,Qon,Rpeak,Qoff,Ton,Toff,Pon2])
                # Labelling untuk 7 kelas dari Pon ke Pon2
                for idx in range(len(Anotasi)-1):
                    start = Anotasi[idx]-Anotasi[0] # contoh start = pon
                    stop = Anotasi[idx+1]-Anotasi[0] # stop = poff
                    label[idx][start:stop] = 1 # dilabelin bos.
                # label[5][Toff-Pon] = 1 # label terakhir di isi 1.
                
                """
                Tambahkan semua sinyal dan label ke dalam satu list
                """
                
                label_transpose = np.transpose(label)
                all_label.append(label_transpose)
                all_sinyal.append(Sinyal_satu_beat)
            else:
                tanda = 0
 
"""
Proses padding zero untuk ukuran semua sinyal dan label satu beat yang kurang dari ukuran maksimal satu beat.
"""
n_kelas = 8
for i in range(len(all_sinyal)):
    print("padding sinyal ke ",i)
    ukuran_beat = all_sinyal[i].shape[0]
    if  ukuran_beat < nilai_max:
        ukuran_padding = nilai_max - ukuran_beat
        padding_beat = np.zeros(ukuran_padding,dtype='int')
        padding_label = np.zeros((ukuran_padding,n_kelas),dtype='int')
        padding_label[:,n_kelas-1] = 1
        all_sinyal_ukuran_sama.extend(np.concatenate((all_sinyal[i],padding_beat)))
        all_label_ukuran_sama.extend(np.concatenate((all_label[i],padding_label)))
    else:
        all_sinyal_ukuran_sama.extend(all_sinyal[i])
        all_label_ukuran_sama.extend(all_label[i])

# MENJADIKAN LIST KE BENTUK ARRAY
all_labels_arr = np.array(all_label_ukuran_sama)
all_sinyal_arr = np.array(all_sinyal_ukuran_sama)


# RESHAPE DATA SINYAL DAN LABEL KE BENTUK 3 DIMENSI (ukuransinyal,timestep,kelas) untuk model lstm
Nilai_satu_detik = nilai_max
s = int(all_sinyal_arr.shape[0] / nilai_max)
all_sinyal_rnn = all_sinyal_arr.reshape((s,nilai_max,1))
all_labels_rnn = all_labels_arr.reshape((s,nilai_max,n_kelas))


def islandinfo(y, trigger_val, stopind_inclusive=True):
    # Setup "sentients" on either sides to make sure we have setup
    # "ramps" to catch the start and stop for the edge islands
    # (left-most and right-most islands) respectively
    y_ext = np.r_[False,y==trigger_val, False]

    # Get indices of shifts, which represent the start and stop indices
    idx = np.flatnonzero(y_ext[:-1] != y_ext[1:])

    # Lengths of islands if needed
    lens = idx[1::2] - idx[:-1:2]

    # Using a stepsize of 2 would get us start and stop indices for each island
    return idx[:-1:2], list(zip(idx[:-1:2], idx[1::2]-int(stopind_inclusive))), lens


import matplotlib.pyplot as plt
sinyal = []
sinyal.extend(all_sinyal[0])
sinyal.extend(all_sinyal[1])

sinyal = np.array(sinyal)

# plot sinyal 2 beat pertama
plt.figure()
plt.plot(range(len(sinyal)),sinyal)


# plot sinyal beat 1
plt.figure()
plt.plot(range(len(all_sinyal[0])),all_sinyal[0])

# plot sinyal beat 2
plt.figure()
plt.plot(range(len(all_sinyal[1])),all_sinyal[1])



#split 90 10
train_data,test_data,train_label, test_label = train_test_split(all_sinyal_rnn,all_labels_rnn,test_size=0.1,random_state=42)

save_file('pickles2/train data 8 kelas',train_data)
save_file('pickles2/test data 8 kelas',test_data)
save_file('pickles2/train label 8 kelas',train_label)
save_file('pickles2/test label 8 kelas',test_label)
save_file('pickles2/semua normal ludb 8 kelas',all_sinyal_rnn)
save_file('pickles2/semua label normal ludb 8 kelas',all_labels_rnn)
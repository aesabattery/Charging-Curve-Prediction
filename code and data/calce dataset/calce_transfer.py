# -*- coding: utf-8 -*-
"""
Created on Mon Oct 08 2020
Use charging voltage and capacity in 300 mV window 
to predict the entire charging curve.
Use transfer learning to predict the charging curves of batteries in CALCE dataset
"""
from keras.models import Sequential
import numpy as np
from keras.callbacks import ModelCheckpoint
import time
import scipy.io as scio
start_time = time.time()

#%% import data
# training data are from calce36, and testing data are from others
location = 'data/'
curve_cell_1 = np.genfromtxt(location+'CALCE_36.txt',delimiter = ',')
# test dataset
curve_cell_2 = np.genfromtxt(location+'CALCE_35.txt',delimiter = ',')
curve_cell_3 = np.genfromtxt(location+'CALCE_37.txt',delimiter = ',')
curve_cell_4 = np.genfromtxt(location+'CALCE_38.txt',delimiter = ',')

# scale the data according to the nomimal capacities of the oxford and calce batteries
curve_cell_1 = curve_cell_1/1.1*0.74
curve_cell_2 = curve_cell_2/1.1*0.74
curve_cell_3 = curve_cell_3/1.1*0.74
curve_cell_4 = curve_cell_4/1.1*0.74

# downsample the training and test datasets 
curve_cell_1 = curve_cell_1[0:-1:45]
curve_cell_2 = curve_cell_2[0:-1:10]
curve_cell_3 = curve_cell_3[0:-1:10]
curve_cell_4 = curve_cell_4[0:-1:10]


curve_train = [curve_cell_1]
curve_test = [curve_cell_2,curve_cell_3,curve_cell_4]
voltage = np.arange(2.71,4.181,0.01)
#%% compute mean and std based on the training data to standarise the input

entire_charge = curve_train[0].flatten()
for ind in range(1,len(curve_train),1):
    entire_charge = np.append(entire_charge,curve_train[ind].flatten())
    
entire_voltage = np.tile(voltage,len(entire_charge)//len(voltage))

entire_series_stack = np.vstack((entire_voltage, entire_charge))
entire_series = entire_series_stack.T
print(entire_charge.shape)
print(entire_voltage.shape)
print(entire_series.shape)
# mean and std 
mean = entire_series.mean(axis=0)
entire_series -= mean
std = entire_series.std(axis=0)
entire_series /= std
#%% 数据准备的函数
def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=1):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            # i += 1
            i += len(rows)
        samples = np.zeros((len(rows),
                           lookback // step-step,
                           data.shape[-1]))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices][1:,:]
            samples[j][:,1] -= data[indices][0,1]
        return samples
#%% prepare data for each cell and form the training and test dataset 
lookback_size = 31 # window size
step_size = 1      # sampling step

# training data
data_train_temp = []
target_train_temp = []
for ind in range(0,len(curve_train),1):
    for k in range(0,len(curve_train[ind]),1):
        charge = curve_train[ind][k]
        temp_train_vstack = np.vstack((voltage, charge))
        temp_train_not = temp_train_vstack.T # not standarisation
        # standarisation
        temp_train = temp_train_not - mean
        temp_train = temp_train/std
        batch_size_train = len(temp_train)
        (train_gen) = generator(temp_train,
                              lookback=lookback_size,
                              delay=0,
                              min_index=0,
                              max_index=None,
                              shuffle=False,
                              batch_size=batch_size_train, 
                              step=step_size)
        data_train_temp.append(train_gen)
        A = np.tile(charge,[len(train_gen),1])
        target_train_temp.append(A)
        
train_gen_final = np.concatenate(data_train_temp,axis=0)
train_target_final = np.concatenate(target_train_temp,axis=0)
print(train_gen_final.shape)
print(train_target_final.shape)



#%% test data
data_test_temp = []
target_test_temp = []
for ind in range(0,len(curve_test),1):
    for k in range(0,len(curve_test[ind]),1):
        charge = curve_test[ind][k] 
        temp_test_vstack = np.vstack((voltage, charge))
        temp_test_not = temp_test_vstack.T
        # standarisation
        temp_test = temp_test_not - mean
        temp_test = temp_test/std
        batch_size_test = len(temp_test)
        (test_gen) = generator(temp_test,
                              lookback=lookback_size,
                              delay=0,
                              min_index=0,
                              max_index=None,
                              shuffle=False,
                              batch_size=batch_size_test, 
                              step=step_size)
        data_test_temp.append(test_gen)
        A = np.tile(charge,[len(test_gen),1])
        target_test_temp.append(A)
        
test_gen_final = np.concatenate(data_test_temp,axis=0)
test_target_final = np.concatenate(target_test_temp,axis=0)
print(test_gen_final.shape)
print(test_target_final.shape)

#%% shuffle the training dataset for validation 
index = np.arange(train_gen_final.shape[0])
np.random.shuffle(index)
 
Input_train = train_gen_final[index,:,:]
Output_train = train_target_final[index,:]

Input_test = test_gen_final
Output_test = test_target_final
#%% import the pretrained model and modify it
from keras import layers
from keras.models import load_model
model_base = load_model('transferbasis/oxford_model.hdf5')
model_base.summary()

model = Sequential()
for layer in model_base.layers[:-1]: # go through until last layer
    model.add(layer)
model.add(layers.Dense(len(voltage),name='new_dense'))
model.summary()

for layer in model.layers[:-1]:
    layer.trainable = False
for i,layer in enumerate(model.layers):
    print(i,layer.name,layer.trainable)
    
model.compile(loss='mean_squared_error', optimizer='adam')

# the number of epochs is set to 50 as a fast example
filepath="models/transfer-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
mode='auto')
callbacks_list = [checkpoint]
# freeze copied layers and train for 50 times 
history = model.fit(Input_train,Output_train,
                    epochs=50,
                    batch_size=512,
                    validation_split=0.35,
                    callbacks=callbacks_list, verbose=1)

for layer in model.layers[:-1]:
    layer.trainable = True
for i,layer in enumerate(model.layers):
    print(i,layer.name,layer.trainable)
# make all layers trainable and train for next 4950 epochs
# the number of epochs is set to 50 as a fast example
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(Input_train,Output_train,
                    epochs=50,
                    batch_size=512,
                    validation_split=0.35,
                    callbacks=callbacks_list, verbose=1)
#%% show training and validation loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

import matplotlib.pyplot as plt
plt.figure(dpi=150)
plt.plot(epochs[1:], np.log(loss[1:]), 'bo', label='Training loss')
plt.plot(epochs[1:], np.log(val_loss[1:]), 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.ylabel('log(loss)')
plt.xlabel('Epoch')
plt.legend()
plt.show()
print("--- %.2s seconds ---" % (time.time() - start_time))





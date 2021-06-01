# -*- coding: utf-8 -*-
"""
Created on Mon Oct 08 2020
Use charging voltage and capacity in 300 mV window 
to predict the entire charging curve.
The oxford dataset is adopted.

"""
from keras.models import Sequential
import numpy as np
from keras.callbacks import ModelCheckpoint
import time
import scipy.io as scio
start_time = time.time()
# load data
#%% import data: cells 1~6 for training and 7~8 for testing
# each row denotes the capacity at 2.8 to 4.19 V at a step of 0.01V
# training dataset
location = 'data/'
curve_cell_1 = np.genfromtxt(location+'q_curve_28_419_cell_1.txt',delimiter = ',')
curve_cell_2 = np.genfromtxt(location+'q_curve_28_419_cell_2.txt',delimiter = ',')
curve_cell_3 = np.genfromtxt(location+'q_curve_28_419_cell_3.txt',delimiter = ',')
curve_cell_4 = np.genfromtxt(location+'q_curve_28_419_cell_4.txt',delimiter = ',')
curve_cell_5 = np.genfromtxt(location+'q_curve_28_419_cell_5.txt',delimiter = ',')
curve_cell_6 = np.genfromtxt(location+'q_curve_28_419_cell_6.txt',delimiter = ',')
curve_train = [curve_cell_1,curve_cell_2,curve_cell_3,curve_cell_4,
               curve_cell_5,curve_cell_6]
# test dataset
curve_cell_7 = np.genfromtxt(location+'q_curve_28_419_cell_7.txt',delimiter = ',')
curve_cell_8 = np.genfromtxt(location+'q_curve_28_419_cell_8.txt',delimiter = ',')
curve_test = [curve_cell_7,curve_cell_8]
voltage = np.arange(2.8,4.19,0.01)
#%% compute mean and std based on the training data to standarise the input
entire_charge = curve_train[0].flatten()
for ind in range(1,len(curve_train),1):
    entire_charge = np.append(entire_charge,curve_train[ind].flatten())
    
entire_voltage = np.tile(voltage,len(entire_charge)//len(voltage))

entire_series_stack = np.vstack((entire_voltage, entire_charge))
entire_series = entire_series_stack.T
mean = entire_series.mean(axis=0)
entire_series -= mean
std = entire_series.std(axis=0)
entire_series /= std
#%% sequence generation function
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
lookback_size = 31 # window
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

#%% shuffle the training dataset 
index = np.arange(train_gen_final.shape[0])
np.random.shuffle(index)
Input_train = train_gen_final[index,:,:]
Output_train = train_target_final[index,:]

Input_test = test_gen_final
Output_test = test_target_final
#%% load the trained model
from keras.models import load_model
# a pretrained model is used here as an example
# change the path where the model saved
model = load_model('models/example_trained_dnn.hdf5') 
#%% make predictions
y_proba = model.predict(Input_test) # make predictions
original_input_sample = Input_test
original_input_sample = original_input_sample*std
original_input_sample += mean
#%% normalise the error of maximum capacity using norminal capacity
cap_error = 100*(y_proba[:,-1] - Output_test[:,-1])/(0.74*3600) 

#%% maximum energy capacity
energy_curve = y_proba*0
energy_curve_ground = y_proba*0

for i in range(0,len(y_proba),1):
    for j in range(0,len(y_proba[1]),1):
        energy_curve[i,j] = np.trapz(voltage[:j+1], y_proba[i,:j+1]/3600)
        energy_curve_ground[i,j]  = np.trapz(voltage[:j+1], Output_test[i,:j+1]/3600)

en_cap_error = 100*(energy_curve[:,-1] - energy_curve_ground[:,-1])/energy_curve_ground[0,-1] # the initial energy of the first cell

#%% save results to a given path
np.savetxt("y_proba.txt", y_proba, delimiter=",")
np.savetxt("Output_test.txt", Output_test, delimiter=",")
np.savetxt("cap_error.txt", cap_error, delimiter=",")
np.savetxt("en_cap_error.txt", en_cap_error, delimiter=",")
print("--- %s seconds ---" % (time.time() - start_time))














   
    
    

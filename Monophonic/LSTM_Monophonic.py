
import numpy as np
import shelve
#from pandas import read_csv, get_dummies
import math
import random
import keras.callbacks
import tensorflow as tf
from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, TimeDistributed, Masking
from keras.optimizers import RMSprop, Adam
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.metrics import mean_squared_error
#from neupy.datasets import make_reber
from keras.utils import np_utils, plot_model
from Monophonic.utils import *
#import matplotlib.pyplot as plt


# Week 9 or smth: 
# Add LSTM layer
# Add next duration as input for learning of the next pitch
# Add Tensorboard
# Use screen -r to keep code running on cluster
# Check Docker ? ? 
# Prototype: with 10 percent of data
# Me:
# Implement twin
# If time or necessary now: Do functional api
# Generate from prediction and create midi file
# Either normalize data to Cmaj/Aminor or augment data to random pitch at every epoch
# Do full crossvalidation in the end


#NAAN FROM FULL MASK INPUT?!?
data_percentage = 0.1

max_song_size = 300 + 1 #(based on histogram) (extra 1 since we roll Y and delete last column)
n_step = 5
step_size = int(max_song_size / n_step) # 100 ~ upper limit
test_percentage = 0.1 # for number of test batches

N_Epoch = 20

add_time = True

print("loading songs")

data, dictionaries = load("BobSturm.pkl")

pitch_indices = dict((p,i) for i, p in enumerate(dictionaries["pitchseqs"]))
pitch_data = np.array(data["pitchseqs"][:int(data_percentage*len(data["pitchseqs"]))])

song_sizes = np.array([len(i) for i in pitch_data]) #do a plt.hist(song_sizes) plt.show()
remove_idx = np.where(song_sizes>max_song_size)

song_sizes = np.delete(song_sizes, remove_idx)
pitch_data = np.delete(pitch_data, remove_idx)

batch_possible = [ k for k in range(1,int(pitch_data.size/10)) if pitch_data.size%k == 0]

batch_size = int(batch_possible[-1])
n_batch = int(pitch_data.size/batch_size)

n_test_batch = int(n_batch*test_percentage)

train_idxes = np.array(range(0,(n_batch-n_test_batch)*batch_size))
test_idxes = np.array(range((n_batch-n_test_batch)*batch_size, n_batch*batch_size))

batches = []
next_pitch = []

print('Vectorization...(TODO: optimize matrices memory usage)')

X = -1*np.ones((n_batch*batch_size, step_size*n_step+1, len(pitch_indices)), dtype=np.int) #or float64
for j, pitches in enumerate(pitch_data):
	categorized = np_utils.to_categorical([pitch_indices[p] for p in pitches], num_classes = 47)
	##if truncating and masking: 
	#X[j,:min(len(pitches,),step_size*n_step)] =  categorized[:min(len(pitches),step_size*n_step)]
	# if masking only:
	X[j,:len(pitches,)] =  categorized


Y = np.roll(X,-1,axis=1)
X = np.delete(X,-1,axis=1)
Y = np.delete(Y,-1,axis=1)


# Adding next pitch duration as input:
time_dim = 0
if add_time:
	time_indices = dict((p,i) for i, p in enumerate(dictionaries["Tseqs"]))
	time_dim = len(time_indices)
	time_data =  np.array(data["Tseqs"][:int(data_percentage*len(data["Tseqs"]))])
	time_data = np.delete(time_data, remove_idx)
	
	X2 = -1*np.ones((n_batch*batch_size, step_size*n_step+1, time_dim), dtype=np.int) #or float64
	for j, times in enumerate(time_data):
		categorized = np_utils.to_categorical([time_indices[t] for t in times], num_classes = time_dim )
		##if truncating and masking: 
		#X[j,:min(len(pitches,),step_size*n_step)] =  categorized[:min(len(pitches),step_size*n_step)]
		# if masking only:
		X2[j,:len(times,)] = categorized
	X2 = np.roll(X2,-1,axis=1)
	X2 = np.delete(X2,-1,axis=1)
	X = np.concatenate((X,X2), axis = 2)

# create and fit the LSTM network
print('Building the Model layers')
model = Sequential()
# Careful !  to categorical uses 0 and 1, so invalid value should be smth else like -1: 
model.add(Masking(mask_value=-1., batch_input_shape=(batch_size,step_size,len(pitch_indices)+time_dim)))
model.add(LSTM(80, return_sequences=True, batch_input_shape=(batch_size,step_size,len(pitch_indices)+time_dim),stateful=True))
# possibly add dropout
model.add(LSTM(80, return_sequences=True, stateful=True))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(len(pitch_indices))))
model.add(Activation('softmax'))
optimizer = Adam(clipnorm=1.)# uwe tf.train.AdamOptimizer()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])

# create a Tensorboard callback
hyperparameter = 'test1'
callback = TensorBoard('tensorboard_logs/'+hyperparameter)
callback.set_model(model)
# >tensorboard.exe --logdir=C:/Users/NiWa/PycharmProjects/ArtificialComposer/Monophonic/tensorboard_logs

# write a log into a tensorflow callback
def write_log(callback, names, logs, batch_no):
	for name, value in zip(names, logs):
		summary = tf.Summary()
		summary_value = summary.value.add()
		summary_value.simple_value = value
		summary_value.tag = name
		callback.writer.add_summary(summary, batch_no)
		callback.writer.flush()

metrics_train = np.zeros((N_Epoch,len(model.metrics_names),n_batch-n_test_batch))
metrics_test = np.zeros((N_Epoch,len(model.metrics_names),n_test_batch))
# plot_model(model,to_file='model_test.png') # I get a unfixable bug : try later

print('Starting to learn:')
for i in range(N_Epoch):
	print('------- {} out of {} Epoch -----'.format(i+1,N_Epoch))

	## Epochs should take all data; batches presented random, reset at each end of batch_size
	np.random.shuffle(train_idxes)
	batches_idxes = np.reshape(train_idxes, (-1,batch_size))
	for j, batch  in enumerate(batches_idxes):
		print('batch {} of {}'.format(j+1,n_batch-n_test_batch))
		for k in range(n_step):
			metrics_train[i, :, j] += model.train_on_batch(X[batch,k*step_size:(k+1)*(step_size)], Y[batch,k*step_size:(k+1)*step_size]) #python 0:3 gives 0,1,2 (which is not intuitive at all)
			write_log(callback, ['train_'+s for s in model.metrics_names], metrics_train[i,:,j], j*n_step+k)
			#metrics_train[i,:,j] += logs #model.train_on_batch(X[batch,k*step_size:(k+1)*(step_size)], Y[batch,k*step_size:(k+1)*step_size]) #python 0:3 gives 0,1,2 (which is not intuitive at all)
		model.reset_states() 

	test_batch_idxes = np.reshape(test_idxes,(-1,batch_size))
	for j, test_batch in enumerate(test_batch_idxes):
		for k in range(n_step):
			metrics_test[i, :, j] += model.test_on_batch(X[test_batch,k*step_size:(k+1)*(step_size)], Y[test_batch,k*step_size:(k+1)*step_size])
			write_log(callback, ['test_'+s for s in model.metrics_names], metrics_test[i, :, j], j * n_step + k)
			#metrics_train[i, :, j] += logs
		model.reset_states()

	metrics_test[i] = metrics_test[i]/float(n_step) # divide only i indice, else division would be done for all at each epoch
	metrics_train[i] = metrics_train[i]/float(n_step)

	print('Train results:\t {} \n \t {}'.format(model.metrics_names, np.mean(metrics_train[i],axis=1))) # mean function just for printing
	print('Test results:\t {} \n \t {}'.format(model.metrics_names, np.mean(metrics_test[i],axis=1) ))


# save the model:
model.save('monoPhonic_model3.h5')
#np.save('monoXdata.npy',X) #too big ?!
#np.save('monoYdata.npy',Y)
np.save('monoTrainMetrics3.npy', metrics_train)
np.save('monoTestMetrics3.npy', metrics_test)


import numpy as np
import shelve
#from pandas import read_csv, get_dummies
import math
import random
import keras.callbacks
#import tensorflow as tf
from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, TimeDistributed, Masking
from keras.optimizers import RMSprop, Adam
from keras.utils import np_utils, plot_model
from utils import *
#import matplotlib.pyplot as plt



#NAAN FROM FULL MASK INPUT?!?
def train_backwards( X, Y, n_batch, n_test_batch, extension = '_', n_step = 5, N_Epoch = 20, LSTM_Size = 80, Layers = 2):

	batch_size = int(X.shape[0]/n_batch)
	song_sizes = X.shape[1]
	step_size = int(song_sizes/n_step)
	pitch_size = Y.shape[2]
	pitch_indices = [i for i in range(pitch_size)]
	time_dim = X.shape[2] - pitch_size 


	train_idxes = np.array(range(0,(n_batch-n_test_batch)*batch_size))
	test_idxes = np.array(range((n_batch-n_test_batch)*batch_size, n_batch*batch_size))

	# create and fit the LSTM network
	print('Building the Model layers')
	model = Sequential()
	# Careful !  to categorical uses 0 and 1, so invalid value should be smth else like -1: 
	model.add(Masking(mask_value= -1., batch_input_shape=(batch_size,step_size,len(pitch_indices)+time_dim)))
	model.add(LSTM(LSTM_Size, return_sequences=True, batch_input_shape=(batch_size,step_size,len(pitch_indices)+time_dim),stateful=True))
	# possibly add dropout
	while(Layers-1):
		model.add(LSTM(LSTM_Size, return_sequences = True, stateful = True))
		Layers -= 1
	model.add(Dropout(0.2))
	model.add(TimeDistributed(Dense(len(pitch_indices))))
	model.add(Activation('softmax'))
	optimizer = Adam(clipnorm=1.)# uwe tf.train.AdamOptimizer()
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])

	# create a Tensorboard callback
	#hyperparameter = 'backward_single'
	#callback = TensorBoard('tensorboard_logs/'+hyperparameter)
	#callback.set_model(model)
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
				#write_log(callback, ['train_'+s for s in model.metrics_names], metrics_train[i,:,j], (j*n_step+k) + (i * len(batch)))
				#metrics_train[i,:,j] += logs #model.train_on_batch(X[batch,k*step_size:(k+1)*(step_size)], Y[batch,k*step_size:(k+1)*step_size]) #python 0:3 gives 0,1,2 (which is not intuitive at all)
			model.reset_states() 

		test_batch_idxes = np.reshape(test_idxes,(-1,batch_size))
		for j, test_batch in enumerate(test_batch_idxes):
			for k in range(n_step):
				metrics_test[i, :, j] += model.test_on_batch(X[test_batch,k*step_size:(k+1)*(step_size)], Y[test_batch,k*step_size:(k+1)*step_size])
				#write_log(callback, ['test_'+s for s in model.metrics_names], metrics_test[i, :, j], (j*n_step+k) + (i * len(test_batch)))
				#metrics_train[i, :, j] += logs
			model.reset_states()

		metrics_test[i] = metrics_test[i]/float(n_step) # divide only i indice, else division would be done for all at each epoch
		metrics_train[i] = metrics_train[i]/float(n_step)

		print('Train results:\t {} \n \t {}'.format(model.metrics_names, np.mean(metrics_train[i],axis=1))) # mean function just for printing
		print('Test results:\t {} \n \t {}'.format(model.metrics_names, np.mean(metrics_test[i],axis=1) ))
	

	#save the model:
	model.save('backward_model_{}.h5'.format(extension))
	#np.save('monoXdata.npy',X) #too big ?!
	#np.save('monoYdata.npy',Y)
	np.save('backwardTrainMetrics_{}.npy'.format(extension), metrics_train)
	np.save('backwardTestMetrics_{}.npy'.format(extension), metrics_test)

	return model



import numpy as np
import shelve
#from pandas import read_csv, get_dummies
import math
import random
import keras.callbacks
#import tensorflow as tf
from keras.models import Model,Sequential
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.layers import Input,Dense, Activation, Dropout
from keras.layers import LSTM, TimeDistributed, Masking
from keras.optimizers import RMSprop, Adam
from keras.utils import np_utils, plot_model
from utils import *
from backward_LSTM import *
from input_manager import * 
#import matplotlib.pyplot as plt

#import theano
#theano.config.openmp = True
#theano.config.cxx = "/opt/local/bin/g++"


import keras.backend as K





#Week qqchose;
# Implement twin: learn backwards first (mirror of forward), use loss from paper for forward (try sequential)
# look for the affine transformation ?

# Week autrechose:
# Finir model => train locally
# Le rapport : Nils ? pour Reber 
# Lancer sur cluster
# Si besoin: Fix mask batch
# Si temps: Cell state comparaison ( = backward devient Functional ++ outputs)



# From : https://github.com/philipperemy/keras-visualize-activations/blob/master/read_activations.py

def get_activations(model, model_inputs, print_shape_only=False, layer_name=None):
    print('----- activations -----')
    activations = []
    inp = model.input

    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs

    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(0.)
    else:
        list_inputs = [model_inputs, 0.]

    # Learning phase. 0 = Test mode (no dropout or batch normalization)
    # layer_outputs = [func([model_inputs, 0.])[0] for func in funcs]
    layer_outputs = [func(list_inputs)[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations



def forward_loss(y_true, y_pred, hidden_weights, model): #TODO:
	l2_norm = 0
	return K.categorical_crossentropy(y_pred,y_true) - l2_norm


def train_forwards(back_model, X, Y,  n_batch, n_test_batch, extension = '_', n_step = 5, N_Epoch = 20, LSTM_Size = 80, Layers = 2):

	batch_size = int(X.shape[0]/n_batch)
	song_sizes = X.shape[1]
	step_size = int(song_sizes/n_step)
	pitch_size = Y.shape[2]
	pitch_indices = [i for i in range(pitch_size)]
	time_dim = X.shape[2] - pitch_size 

	train_idxes = np.array(range(0,(n_batch-n_test_batch)*batch_size))
	test_idxes = np.array(range((n_batch-n_test_batch)*batch_size, n_batch*batch_size))

	print('Building the Forward Model layers')

	main_input= Input(batch_shape=(batch_size,step_size,len(pitch_indices)+time_dim))
    mask = Masking(mask_value = -1, batch_input_shape=(batch_size,step_size,len(pitch_indices)+time_dim))(main_input)
    lstm_1 = LSTM(LSTM_Size, return_sequences=True, batch_input_shape=(batch_size,step_size,len(pitch_indices)+time_dim),stateful=True)(mask)
	lstm_2_out = LSTM(LSTM_Size, return_sequences = True, stateful = True, name = "lstm_2_out")(lstm_1)
    dropout = Dropout(0.2)(lstm_2_out)
    dense = TimeDistributed(Dense(len(pitch_indices)))(dropout)
    main_output = Activation('softmax', name= "main_output")(dense)

	model = Model(inputs = [main_input], outputs = [main_output, lstm_2_out])
	optimizer =  Adam(clipnorm=1.)
	model.compile(optimizer = optimizer, metrics = ["accuracy"],
		loss = {'main_output': 'categorical_crossentropy', 'lstm_2_out': 'mean_squared_error'},
		loss_weights={'main_output': 1., 'lstm_2_out': 0.5})

#	model = Sequential()
#	# Careful !  to categorical uses 0 and 1, so invalid value should be smth else like -1: 
#	model.add(Masking(mask_value= -1., batch_input_shape=(batch_size,step_size,len(pitch_indices)+time_dim)))
#	model.add(LSTM(80, return_sequences=True, batch_input_shape=(batch_size,step_size,len(pitch_indices)+time_dim),stateful=True))
#	# possibly add dropout
#	model.add(LSTM(80, return_sequences = True, stateful = True))
#	model.add(Dropout(0.2))
#	model.add(TimeDistributed(Dense(len(pitch_indices))))
#	model.add(Activation('softmax'))
#	optimizer = Adam(clipnorm=1.)# uwe tf.train.AdamOptimizer()
#	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])

	# create a Tensorboard callback
	#hyperparameter = 'forward_single'
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
	while(len(back_model.layers) != 3):
		back_model.pop() #getting to hidden layer lstm
	# popped activation, dense and dropout
	for i in range(N_Epoch):
		print('------- {} out of {} Epoch -----'.format(i+1,N_Epoch))

		## Epochs should take all data; batches presented random, reset at each end of batch_size
		np.random.shuffle(train_idxes)
		batches_idxes = np.reshape(train_idxes, (-1,batch_size))
		for j, batch  in enumerate(batches_idxes):
			print('batch {} of {}'.format(j+1,n_batch-n_test_batch))
			hidden_values = []
			print(' Getting backward weights:')
			for k in range(n_step):
				X_reversed = np.flip(X[batch,(n_step-k-1)*step_size:(n_step-k)*(step_size),:len(pitch_indices)],axis = 1) # start with end of song
				hidden_values.append( np.flip(back_model.predict(X_reversed, batch_size = len(batch)), axis = 1) ) # flip back prediction to song order
			print(' Training Forward Model')
			for k in range(n_step):
				metrics_train[i, :, j] += model.train_on_batch(
					X[batch,k*step_size:(k+1)*(step_size)],
					{'main_output': Y[batch,k*step_size:(k+1)*step_size], 'lstm_2_out': hidden_values[n_step-k-1]}) #last hidden value batch is first song part
				#write_log(callback, ['train_'+s for s in model.metrics_names], metrics_train[i,:,j]/(k+1), (j*n_step+k) + (i * len(batch)))
				#metrics_train[i,:,j] += logs #model.train_on_batch(X[batch,k*step_size:(k+1)*(step_size)], Y[batch,k*step_size:(k+1)*step_size]) #python 0:3 gives 0,1,2 (which is not intuitive at all)
			model.reset_states() 

		test_batch_idxes = np.reshape(test_idxes,(-1,batch_size))
		for j, test_batch in enumerate(test_batch_idxes):
			hidden_test_values = [] 
			for k in range(n_step):
				X_reversed = np.flip(X[test_batch,(n_step-k-1)*step_size:(n_step-k)*(step_size),:len(pitch_indices)],axis = 1) # start with end of song
				hidden_test_values.append(np.flip(back_model.predict(X_reversed, batch_size = len(batch)), axis = 1))
			for k in range(n_step):
				metrics_test[i, :, j] += model.test_on_batch(
					X[test_batch,k*step_size:(k+1)*(step_size)],
					{'main_output': Y[test_batch,k*step_size:(k+1)*step_size], 'lstm_2_out': hidden_test_values[n_step-k-1]}) #last hidden value batch is first song part
				#write_log(callback, ['test_'+s for s in model.metrics_names], metrics_test[i, :, j]/(k+1), (j*n_step+k) + (i * len(test_batch)) )
				#metrics_train[i, :, j] += logs
			model.reset_states()

		metrics_test[i] = metrics_test[i]/float(n_step) # divide only i indice, else division would be done for all at each epoch
		metrics_train[i] = metrics_train[i]/float(n_step)

		print('Train results:\t {} \n \t {}'.format(model.metrics_names, np.mean(metrics_train[i],axis=1))) # mean function just for printing
		print('Test results:\t {} \n \t {}'.format(model.metrics_names, np.mean(metrics_test[i],axis=1) ))
	

	#save the model:
	model.save('forward_model_{}.h5'.format(extension))
	#np.save('monoXdata.npy',X) #too big ?!
	#np.save('monoYdata.npy',Y)
	np.save('forwardTrainMetrics_{}.npy'.format(extension), metrics_train)
	np.save('forwardTestMetrics_{}.npy'.format(extension), metrics_test)

	return 0











if __name__ == "__main__":
	#Default: song_sizes = 300, percentage = 0.1, add_time = False, N_Epoch = 20, LSTM_Size = 40, Minimum_batches = 10
	load = False
	name = '120_1Layer_300_full'
	if load:
		print('Loading backward model')
		back_model = load_model("backward_model_{}.h5".format(name))
	else:
		print('Get Reversed train input')
		X_rev, Y_rev, n_batch, n_test_batch = get_model_input(reverse = True, song_sizes = 300, Minimum_batches = 15, percentage = 1)
		print('callig backward model')
		back_model = train_backwards(X_rev, Y_rev, n_batch, n_test_batch, extension = name, LSTM_Size = 120, Layers = 1) # song size 350, min_batch = 5
	print('Get forward input')
	X,Y, n_batch, n_test_batch = get_model_input(add_time = True, song_sizes = 300, Minimum_batches = 15, percentage = 1)
	train_forwards(back_model, X, Y, n_batch, n_test_batch, extension = name, LSTM_Size  = 120, Layers = 1)




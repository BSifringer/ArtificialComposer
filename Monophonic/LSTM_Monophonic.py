
# LSTM for Embedded Continuous Reber Grammar 
import numpy as np
import shelve
from pandas import read_csv, get_dummies
import math
import random
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, TimeDistributed, Masking
from keras.optimizers import RMSprop
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from neupy.datasets import make_reber
from keras.utils import np_utils, plot_model
from utils import *


#step_size = 50
#n_step=2 #redefine this. IT is to make 1000 reber words#

#batch_size = 10
#n_batch = 15 # total number
#n_test_batch = 2 # testing part#

#train_idxes = np.array(range(0,(n_batch-n_test_batch)*batch_size))
#test_idxes = np.array(range((n_batch-n_test_batch)*batch_size, n_batch*batch_size))#

#N_Epoch = 80#
#

#print('Making dataset of {} chars * {} steps, with {} times batches of {}'.\
#									format(step_size,n_step,100,batch_size))#

#D=[]
#for i in range(n_batch*batch_size):
#	dataset=''
#	while len(dataset) < step_size*n_step+1 - 0.1*step_size*n_step: 
#		embeddedStep = random.choice('TP')
#		custom = 'P'
#		if(embeddedStep == 'P'):
#			custom = 'X'
#		#dataset += 'B' + embeddedStep + 'B' + make_reber(1)[0] + custom + 'E' + embeddedStep + 'E'
#		dataset += 'B' + embeddedStep + 'B' + make_reber(1)[0] + 'E' + embeddedStep + 'E'
#	D.append(dataset)


max_song_size = 500 + 1 #(based on histogram) (extra 1 since we roll Y and delete last column)
n_step = 5
step_size = int(max_song_size / n_step)
test_percentage = 0.1 # for number of test batches

N_Epoch = 5

print("loading songs")
	
data, dictionaries = load("BobSturm.pkl")



#chars = sorted(list(set(D[0])))
#char_indices = dict((c, i) for i, c in enumerate(chars))
#print(char_indices)
pitch_indices = dict((p,i) for i, p in enumerate(dictionaries["pitchseqs"]))
pitch_data = np.array(data["pitchseqs"])

song_sizes = np.array([len(i) for i in data["pitchseqs"]]) #do a plt.hist(song_sizes)
remove_idx = np.where(song_sizes>max_song_size)

song_sizes = np.delete(song_sizes, remove_idx)
pitch_data = np.delete(pitch_data, remove_idx)

batch_possible = [ k for k in range(1,int(pitch_data.size/50)) if pitch_data.size%k == 0]

batch_size = int(batch_possible[-1])
n_batch = int(pitch_data.size/batch_size)

n_test_batch = int(n_batch*test_percentage)

train_idxes = np.array(range(0,(n_batch-n_test_batch)*batch_size))
test_idxes = np.array(range((n_batch-n_test_batch)*batch_size, n_batch*batch_size))

batches = []
next_pitch = []

print('Vectorization...(TODO: optimize matrices memory usage)')
#X = np.zeros((n_batch*batch_size, step_size*n_step+1, len(chars)), dtype=np.bool)
#for j, word in enumerate(D):
#    for i in range(step_size*n_step+1):
#    	if(i<len(word)):
#	        X[j, i, char_indices[word[i]]] = 1

X = -1*np.zeros((n_batch*batch_size, step_size*n_step+1, len(pitch_indices)), dtype=np.int) #or float64
for j, pitches in enumerate(pitch_data):
	categorized = np_utils.to_categorical([pitch_indices[p] for p in pitches], num_classes = 47)
	##if truncating and masking: 
	#X[j,:min(len(pitches,),step_size*n_step)] =  categorized[:min(len(pitches),step_size*n_step)]
	# if masking only:
	X[j,:len(pitches,)] =  categorized
	#X[j] = np_utils.to_categorical(word, len(chars))


Y=np.roll(X,-1,axis=1)
X = np.delete(X,-1,axis=1)
Y = np.delete(Y,-1,axis=1)

# create and fit the LSTM network
print('Building the Model layers')
model = Sequential()
# Careful !  to categorical uses 0 and 1, so invalid value should be smth else like -1: 
model.add(Masking(mask_value= 0, batch_input_shape=(batch_size,step_size,len(pitch_indices))))
model.add(LSTM(80, return_sequences=True, batch_input_shape=(batch_size,step_size,len(pitch_indices)),stateful=True))
#model.add(LSTM(80,input_shape=(n_step*step_size,len(chars)), activation ='sigmoid',inner_activation = 'hard_sigmoid', return_sequences=True))
#model.add(LSTM(64,return_sequences=True,stateful=True))
model.add(Dropout(0.2))

model.add(TimeDistributed(Dense(len(pitch_indices))))
#model.add(TimeDistributed(Dense(10)))
#model.add(TimeDistributed(Dense(7)))
model.add(Activation('softmax'))
#model.add(Activation('sigmoid'))
optimizer = RMSprop(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])


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
			metrics_train[i,:,j] += model.train_on_batch(X[batch,k*step_size:(k+1)*(step_size)], Y[batch,k*step_size:(k+1)*step_size]) #python 0:3 gives 0,1,2 (which is not intuitive at all)
		model.reset_states() 

	test_batch_idxes = np.reshape(test_idxes,(-1,batch_size))
	for j, test_batch in enumerate(test_batch_idxes):
		for k in range(n_step):
			metrics_test[i,:,j] += model.test_on_batch(X[test_batch,k*step_size:(k+1)*(step_size)], Y[test_batch,k*step_size:(k+1)*step_size])
		model.reset_states() 

	metrics_test[i] = metrics_test[i]/float(n_step) # divide only i indice, else division would be done for all at each epoch
	metrics_train[i] = metrics_train[i]/float(n_step)

	print('Train results:\t {} \n \t {}'.format(model.metrics_names, np.mean(metrics_train[i],axis=1))) # mean function just for printing
	print('Test results:\t {} \n \t {}'.format(model.metrics_names, np.mean(metrics_test[i],axis=1) ))


#model.fit(X, Y, epochs=250, batch_size=batch_size, verbose=2, shuffle=False)


#save the model:
model.save('monoPhonic_model.h5')
#np.save('monoXdata.npy',X) #too big ?!
#np.save('monoYdata.npy',Y)
np.save('monoTrainMetrics.npy', metrics_train)
np.save('monoTestMetrics.npy', metrics_test)


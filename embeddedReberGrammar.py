
# LSTM for Embedded Continuous Reber Grammar 
import numpy as np
import matplotlib.pyplot as plt
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



# Week 3 :
# notes: zero padds asks for a mask to neglect zeros
# We will not padd, we will truncate. 
# Make embedded strings size 100T, T = 50 letters
# Make about 100*batch_size of these and batch size in model fit about 16
# look for to categorical from np.utils
# use roll to have a Y which is a shifted X
# (do this before truncating is a great strategy => y won't all finish with B's)
# Check fit models that are stateful, must forget between batches
#		must not forget between steps


# Week 4: 
# Train like in BachProp5:  
# CHECK TRAIN ON BATCH ! ! 
# Epochs should take all data;
# batches presented random, reset at each end of batch_size
# 10* testData? Should training indexes before 
# keras.utils - plot_model
# plot loss accuracy vs epoch
# plot imshow predict percentage on 1 word


step_size = 50
n_step=100 #redefine this. IT is to make 1000 reber words

batch_size = 16
n_batch = 12 # total number
n_test_batch = 2 # testing part

train_idxes = np.array(range(0,(n_batch-n_test_batch)*batch_size))
test_idxes = np.array(range((n_batch-n_test_batch)*batch_size, n_batch*batch_size))

N_Epoch = 20


print('Making dataset of {} chars * {} steps, with {} times batches of {}'.\
									format(step_size,n_step,100,batch_size))

D=[]
for i in range(n_batch*batch_size):
	dataset=''
	while len(dataset) < step_size*n_step+1 - 20: 
		embeddedStep = random.choice('TP') 
		dataset += 'B' + embeddedStep + make_reber(1)[0] + embeddedStep + 'E'
	D.append(dataset)

#print(len(zip(*D)[0]))
chars = sorted(list(set(D[0])))
char_indices = dict((c, i) for i, c in enumerate(chars))
print(char_indices)

batches = []
next_char = []

print('Vectorization...')
#X = np.zeros((n_batch*batch_size, step_size*n_step+1, len(chars)), dtype=np.bool)
#for j, word in enumerate(D):
#    for i in range(step_size*n_step+1):
#    	if(i<len(word)):
#	        X[j, i, char_indices[word[i]]] = 1

X = -1*np.ones((n_batch*batch_size, step_size*n_step+1, len(chars)), dtype=np.float64)
for j, word in enumerate(D):
	categorized = np_utils.to_categorical([char_indices[c] for c in word])
	X[j,:min(len(word,),step_size*n_step+1)] =  categorized[:min(len(word),step_size*n_step+1)]
	#X[j] = np_utils.to_categorical(word, len(chars))


Y=np.roll(X,-1,axis=1)
X = np.delete(X,-1,axis=1)
Y = np.delete(Y,-1,axis=1)

# create and fit the LSTM network
print('Building the Model layers')
model = Sequential()
# Careful !  to categorical uses 0 and 1, so invalid value should be smth else like -1: 
model.add(Masking(mask_value= -1., batch_input_shape=(batch_size,step_size,len(chars))))
model.add(LSTM(128, return_sequences=True, batch_input_shape=(batch_size,step_size,len(chars)),stateful=True))
#model.add(LSTM(80,input_shape=(n_step*step_size,len(chars)), activation ='sigmoid',inner_activation = 'hard_sigmoid', return_sequences=True))
#model.add(LSTM(64,return_sequences=True,stateful=True))
model.add(Dropout(0.2))

model.add(TimeDistributed(Dense(len(chars))))
#model.add(TimeDistributed(Dense(10)))
#model.add(TimeDistributed(Dense(7)))
model.add(Activation('softmax'))
#model.add(Activation('sigmoid'))
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])


metrics_train = np.zeros((N_Epoch,len(model.metrics_names)))
metrics_test = np.zeros((N_Epoch,len(model.metrics_names)))
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
			metrics_train[i] += model.train_on_batch(X[batch,k*step_size:(k+1)*(step_size)], Y[batch,k*step_size:(k+1)*step_size]) #python 0:3 gives 0,1,2 (which is not intuitive at all)
		model.reset_states() 

	test_batch_idxes = np.reshape(test_idxes,(-1,batch_size))
	for test_batch in test_batch_idxes:
		for k in range(n_step):
			metrics_test[i] += model.test_on_batch(X[test_batch,k*step_size:(k+1)*(step_size)], Y[test_batch,k*step_size:(k+1)*step_size])
		model.reset_states() 

	print('Train results:\t {} \n \t {}'.format(model.metrics_names, metrics_train[i]/(n_batch-n_test_batch) ))
	print('Test results:\t {} \n \t {}'.format(model.metrics_names, metrics_test[i]/(n_test_batch) ))


#model.fit(X, Y, epochs=250, batch_size=batch_size, verbose=2, shuffle=False)


#save the model:
model.save('embedCerg_model.h5')
np.save('embedXdata.npy',X)
np.save('embedydata.npy',Y)
np.save('embedTrainMetrics', metrics_train)
np.save('embedTestMetrics', metrics_test)


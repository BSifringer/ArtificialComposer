
# LSTM for Embedded Continuous Reber Grammar 
import numpy as np
import matplotlib.pyplot as plt
import shelve
from pandas import read_csv
import math
import random
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, TimeDistributed
from keras.optimizers import RMSprop
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from neupy.datasets import make_reber


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


# LSTM for Reber Grammar 
import numpy as np
import matplotlib.pyplot as plt
import shelve
from pandas import read_csv
import math
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from neupy.datasets import make_reber


# notes: zero padds asks for a mask to neglect zeros
# We will not padd
# look for to categorical from np.utils
# use roll to shift y from x
# Check fit models that are stateful, must forget between batches
#		must not forget between steps

step_size = 50
n_step=100 #redefine this. IT is to make 1000 reber words

batch_size = 16
n_batch = 1

iteration=50 #learning



print('Making dataset of {} chars * {} steps, with {} times batches of {}'.\
									format(step_size,n_step,100,batch_size))

D=[]
for i in range(n_batch*batch_size):
	dataset=''
	while len(dataset) < step_size*n_step+1 : 
		embeddedStep = random.choice('TP') 
		dataset += 'B' + embeddedStep + make_reber(1)[0] + embeddedStep + 'E'
	D.append(dataset)

#print(len(zip(*D)[0]))
chars = sorted(list(set(D[0])))
char_indices = dict((c, i) for i, c in enumerate(chars))

batches = []
next_char = []

print('Vectorization...')
X = np.zeros((n_batch*batch_size, step_size*n_step+1, len(chars)), dtype=np.bool)
for j, word in enumerate(D):
    for i in range(step_size*n_step+1):
        X[j, i, char_indices[word[i]]] = 1

Y=np.roll(X,-1,axis=1)
X = np.delete(X,-1,axis=1)
Y = np.delete(Y,-1,axis=1)

# create and fit the LSTM network
print('Building the Model layers')
model = Sequential()
model.add(LSTM(80, return_sequences=True, batch_input_shape=(batch_size,step_size,len(chars)),stateful=True))
#model.add(LSTM(80,input_shape=(n_step*step_size,len(chars)), activation ='sigmoid',inner_activation = 'hard_sigmoid', return_sequences=True))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(len(chars))))
#model.add(TimeDistributed(Dense(10)))
#model.add(TimeDistributed(Dense(7)))
model.add(Activation('softmax'))
#model.add(Activation('sigmoid'))
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])

print('Starting to learn:')
for i in range(iteration):
	print('{} out of {}'.format(i,iteration))
	#check parameters, shuffle True? Stateful False? 

	## CHECK TRAIN ON BATCH ! ! 
	## Epochs should take all data; batches presented random, reset at each end of batch_size
	for j in range(n_step):
		model.fit(X[:,j*step_size:(j+1)*(step_size)], Y[:,j*step_size:(j+1)*step_size], epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
	model.reset_states() 

#model.fit(X, Y, epochs=250, batch_size=batch_size, verbose=2, shuffle=False)


#save the model:
model.save('embedCerg_model.h5')
np.save('embedXdata.npy',X)
np.save('embedydata.npy',Y)


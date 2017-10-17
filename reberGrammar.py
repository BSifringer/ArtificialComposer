# LSTM for Reber Grammar 
import numpy as np
import matplotlib.pyplot as plt
import shelve
from pandas import read_csv
import math
import random
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

batch_size = 20;
n_batch=50; #redefine this. IT is to make 1000 reber words

iteration=50 #learning


dataset = make_reber(batch_size*n_batch)
dataset = ''.join(dataset)
chars = sorted(list(set(dataset)))
char_indices = dict((c, i) for i, c in enumerate(chars))
print('DataSet concatenated lenght:', len(dataset))

batches = []
next_char = []
step = 1
for i in range(0, len(dataset) - batch_size, step):
    batches.append(dataset[i: i + batch_size])
    next_char.append(dataset[i + batch_size])
print('nb batches of size {}:'.format(batch_size), len(batches))


print('Vectorization...')
X = np.zeros((len(batches), batch_size, len(chars)), dtype=np.bool)
y = np.zeros((len(batches), len(chars)), dtype=np.bool)
for j, batch in enumerate(batches):
    for i, data in enumerate(batch):
        X[j, i, char_indices[data]] = 1
    y[j, char_indices[next_char[j]]] = 1


# create and fit the LSTM network
print('Building the Model layers')
model = Sequential()
model.add(LSTM(80, input_shape=(batch_size,len(chars))))
model.add(Dropout(0.2))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
char_indices = dict((c, i) for i, c in enumerate(chars))

print('Starting to learn:')
for i in range(iteration):
	print('{} out of {}'.format(i,iteration))
	#check parameters, shuffle True? Stateful False? 
	model.fit(X, y, epochs=1, batch_size=batch_size, verbose=2, shuffle=True)
	model.reset_states() # What's this? 


#save the model:
model.save('Cerg_model.h5')
np.save('Xdata.npy',X)
np.save('ydata.npy',y)


import shelve
from keras.optimizers import RMSprop
from neupy.datasets import make_reber
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, TimeDistributed
from keras.layers import LSTM
from keras.utils import np_utils, plot_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random
from utils import *


# Do we predict a single batch or a full song? 
# !! Input shape has shape of batch
# For rythm, take a random song? 
# Generate until random song is empty ? => Multiple batches? 
# Idea: never reset states, use predict on single note and rythm and save in vector => no need to work with changing batches

print('Loading model')
if 'model' not in vars():
	model = load_model('monoPhonic_model2.h5')
else:
	model.reset_states()
#TODO: Improve this crappy copy-paste: 
pitch_indices = {48: 0, 49: 1, 50: 2, 51: 3, 52: 4, 53: 5, 54: 6, 55: 7, 56: 8, 57: 9, 58: 10, 59: 11,\
				 60: 12, 61: 13, 62: 14, 63: 15, 64: 16, 65: 17, 66: 18, 67: 19, 68: 20, 69: 21, 70: 22,\
				 71: 23, 72: 24, 73: 25, 74: 26, 75: 27, 76: 28, 77: 29, 78: 30, 79: 31, 80: 32, 81: 33, 82: 34,\
				 83: 35, 84: 36, 85: 37, 86: 38, 87: 39, 88: 40, 89: 41, 90: 42, 91: 43, 92: 44, 93: 45, 94: 46}
indices_pitch = dict((i, p) for i, p in enumerate(pitch_indices))



input_shape = (model.get_layer(index=0)).input_shape

xPredictBatch = -1*np.ones(input_shape)

xPredictBatch[0,0] = np.zeros(len(xPredictBatch[0,0]));

added_time = len(xPredictBatch[0,0])>50


if added_time:
	print(' Vectorizing rythm from random song')
	data, dictionaries = load("BobSturm.pkl")
	time_indices = dict((p,i) for i, p in enumerate(dictionaries["Tseqs"]))
	time_dim = len(time_indices)
	time_data =  np.array(data["Tseqs"])
	random_rythm = random.choice(time_data)
	#xPredictBatch[0,:min(len(random_rythm),len(input_shape[0,:,0])), -time_dim:] = np_utils.to_categorical([time_indices[t] for t in random_rythm], num_classes = time_dim) 

#add 1 in pitch len, and 1 in rest 

print('Predicting')
for i in range(len(xPredictBatch[0])):
	P = model.predict(xPredictBatch, batch_size = input_shape[0])
	choice = np.argmax(P[0,i])

	xPredictBatch[0,i+1,:-time_dim] = np.zeros(len(input_shape[0,0])-time_dim)
	xPredictBatch[0,i+1,choice] = 1
	model.reset_states()

P = P.transpose(0,2,1)

imgplot = plt.imshow(P[0])
plt.yticks(np.arange(P[0].shape[0]), list(indices_char.values()))
plt.xticks(np.arange(P[0].shape[1]), [indices_char[np.argmax(i)] for i in xPredictBatch[0]])
plt.title('Generated Reber Word sampled with argmax')
#plt.colorbar()
plt.show()
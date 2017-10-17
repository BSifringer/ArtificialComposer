from neupy.datasets import make_reber
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
import numpy as np

samples = make_reber(1000)
chars = sorted(list(set(samples)))


concatenated_samples = 'B'+'EB'.join(samples)+'E'

X = concatenated_samples[0:(len(concatenated_samples)-1)]
Y = concatenated_samples[1:len(concatenated_samples)]

chars = sorted(list(set(X)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

Array = np.asarray([char_indices[c] for c in X])

categorized = np_utils.to_categorical([char_indices[c] for c in X])
categorizedY = np_utils.to_categorical([char_indices[c] for c in Y])
decategorized = np.argmax(categorized,axis=1)

decoded = ''.join([indices_char[i] for i in decategorized])

def batchify(X,Y,num_batches,batch_size,batch_length):
	retX = np.ndarray(shape=np.append([num_batches,batch_size,batch_length],X.shape[1:]))
	retY = np.ndarray(shape=np.append([num_batches,batch_size],Y.shape[1:]))
	for i in range(num_batches):
		for j in range(batch_size):
			for k in range(batch_length):
				retX[i][j][k]=X[j+i*batch_length+k]
			retY[i][j]=Y[j+(i+1)*batch_length-1]
	return retX,retY

batch_X,batch_Y = batchify(categorized,categorizedY,10,10,10)

blubb=0 #mark breakpoint to debug ;)
exit(0)
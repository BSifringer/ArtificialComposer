from keras.optimizers import RMSprop
from neupy.datasets import make_reber
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, TimeDistributed
from keras.layers import LSTM
from keras.utils import np_utils, plot_model
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

#Array = np.asarray([char_indices[c] for c in X])

categorized = np_utils.to_categorical([char_indices[c] for c in X])
categorizedY = np_utils.to_categorical([char_indices[c] for c in Y])
decategorized = np.argmax(categorized,axis=1)

decoded = ''.join([indices_char[i] for i in decategorized])

def batchify(X,Y,num_batches,batch_size,batch_length):
    retX = np.ndarray(shape=np.append([num_batches,batch_size,batch_length],X.shape[1:]))
    retY = np.ndarray(shape=np.append([num_batches,batch_size,batch_length],Y.shape[1:]))
    for i in range(num_batches):
        for j in range(batch_size):
            for k in range(batch_length):
                retX[i][j][k]=X[j+i*batch_length+k]
                retY[i][j][k]=Y[j+i*batch_length+k]
    return retX,retY

num_batches = 300
batch_size=20
batch_length=20

batch_X,batch_Y = batchify(categorized,categorizedY,num_batches,batch_size,batch_length)

#debug examples...
#char_X = [char_indices[c] for c in X]
#char_Y = [char_indices[c] for c in Y]
#batch_X,batch_Y = batchify(np.asarray([char_indices[c] for c in X]).reshape(len(X),1),np.asarray([char_indices[c] for c in Y]).reshape(len(Y),1),10,10,10)


# create and fit the LSTM network
print('Building the Model layers')
model = Sequential()
model.add(LSTM(80, return_sequences=True,input_shape=(batch_size,len(chars)),stateful=True,batch_size=batch_size))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(len(chars))))
model.add(Activation('softmax'))
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
char_indices = dict((c, i) for i, c in enumerate(chars))

print('Starting to learn:')
for j in range(5):
    model.reset_states()
    for i in range(num_batches):
        print('{} out of {}, "epoch": {}'.format(i,num_batches,j))
        model.train_on_batch(batch_X[i], batch_Y[i])
        #test on batch for evaluation


model.save("playground_model.h5",overwrite=True)

np.save('playground_Xdata.npy',batch_X)
np.save('playground_Ydata.npy',batch_Y)

blubb=0 #mark breakpoint to debug ;)
exit(0)
from keras.optimizers import RMSprop
from neupy.datasets import make_reber
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
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

#Array = np.asarray([char_indices[c] for c in X])

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
model.add(LSTM(80, input_shape=(batch_size,len(chars)),stateful=True,batch_size=batch_size))
model.add(Dropout(0.2))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
char_indices = dict((c, i) for i, c in enumerate(chars))

print('Starting to learn:')
for j in range(2):
    for i in range(num_batches):
        print('{} out of {}, "epoch": {}'.format(i,num_batches,j))
        model.fit(batch_X[i], batch_Y[i], epochs=1, batch_size=batch_size, verbose=1)
    model.reset_states()

model.evaluate(batch_X[0],batch_Y[0],batch_size)

#compare:
#a=model.predict(batch_X[1],batch_size=batch_size,verbose=1)
#np.argmax(a,axis=1)
#np.argmax(batch_Y[1],axis=1)



#testing area

testsamples = make_reber(1000)

testconcatenated_samples = 'B'+'EB'.join(samples)+'E'

testX = concatenated_samples[0:(len(concatenated_samples)-1)]
testY = concatenated_samples[1:len(concatenated_samples)]

testcategorizedX = np_utils.to_categorical([char_indices[c] for c in X])
testcategorizedY = np_utils.to_categorical([char_indices[c] for c in Y])

testbatch_X,testbatch_Y = batchify(testcategorizedX,testcategorizedY,num_batches,batch_size,batch_length)

correct_predictions=0;
for i in range(num_batches):
    a = model.predict(testbatch_X[i],batch_size)
    correct_predictions=correct_predictions+sum(np.argmax(a,axis=1)==np.argmax(testbatch_Y[i],axis=1))
print('number of correct predictions: {} ({}%)'.format(correct_predictions,(float(correct_predictions)/num_batches/batch_size*100)))

blubb=0 #mark breakpoint to debug ;)
exit(0)
from keras.optimizers import RMSprop
from neupy.datasets import make_reber
from keras.models import Sequential, load_model
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

chars = sorted(list(set(X)))

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

model = load_model("playground_model.h5")
#problem with pydot & graphviz although installed?!
#plot_model(model,"playground_model.png")

#testing area

testsamples = make_reber(1000)

testconcatenated_samples = 'B'+'EB'.join(testsamples)+'E'

testX = testconcatenated_samples[0:(len(testconcatenated_samples)-1)]
testY = testconcatenated_samples[1:len(testconcatenated_samples)]

testcategorizedX = np_utils.to_categorical([char_indices[c] for c in testX])
testcategorizedY = np_utils.to_categorical([char_indices[c] for c in testY])

testbatch_X,testbatch_Y = batchify(testcategorizedX,testcategorizedY,num_batches,batch_size,batch_length)

correct_predictions=0;
for i in range(num_batches):
    a = model.predict(testbatch_X[i],batch_size,verbose=1)
    correct_predictions=correct_predictions+sum(np.argmax(a,axis=2)[0]==np.argmax(testbatch_Y[i],axis=2)[0])
print('number of correct predictions: {} ({}%)'.format(correct_predictions,(float(correct_predictions)/num_batches/batch_size*100)))


blubb=0 #mark breakpoint to debug ;)
exit(0)
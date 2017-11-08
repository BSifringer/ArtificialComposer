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

#reberWord wrong but 50 length is memorized
#reber_word = 'BTBVVPTXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXVPXXXXXXXXXVPXVPXXXXXXVPXXXXXVPXVSETE'
#reber_word = 'BTBVVPTXXXXXXXXXXXXXXXXVPXVPXXXXXXVPXXXXXVPXVSPETE'
#reber_word = 'BTBVVPTXXVSPETE'

model = load_model('embedCerg_model.h5')
train_metrics = np.load('embedTrainMetrics.npy')
test_metrics = np.load('embedTestMetrics.npy')
X = np.load('embedXdata.npy')
char_indices = {'B': 0, 'E': 1, 'P': 2, 'S': 3, 'T': 4, 'V': 5, 'X': 6}
indices_char = dict((i, c) for i, c in enumerate(char_indices))



input_shape = (model.get_layer(index=0)).input_shape

xPredictBatch = X[-input_shape[0]:,:input_shape[1]]
#categorized = np_utils.to_categorical([char_indices[c] for c in reber_word])
#xPredictBatch[0,0:len(reber_word)] = categorized

P = model.predict(xPredictBatch, batch_size = input_shape[0])
model.pop()
model.pop()
H = model.predict(xPredictBatch, batch_size = input_shape[0])


P = P.transpose(0,2,1)
H = H.transpose(0,2,1)

imgplot = plt.imshow(P[0])
plt.yticks(np.arange(P[0].shape[0]), list(indices_char.values()))
plt.xticks(np.arange(P[0].shape[1]), [indices_char[np.argmax(i)] for i in xPredictBatch[0]])
#plt.colorbar()
plt.show()

mean_train_metrics = np.mean(train_metrics,axis=2)
mean_test_metrics = np.mean(test_metrics,axis=2)

std_train_metrics = np.std(train_metrics,axis=2)
std_test_metrics = np.std(test_metrics,axis=2)

x = range(mean_train_metrics.shape[0])



plt.figure()
f, axes = plt.subplots(2,1)
#plt.errorbar(x, mean_test_metrics[:,0], yerr = std_test_metrics[:,0])
axes[1].errorbar(x, mean_train_metrics[:,0],yerr = std_train_metrics[:,0], color = 'red')
axes[1].errorbar(x, mean_test_metrics[:,0], yerr = std_test_metrics[:,0], color = 'blue')
axes[1].set_ylabel('Loss')

axes[0].errorbar(x, mean_train_metrics[:,1], yerr = std_train_metrics[:,1], color = 'red')
axes[0].errorbar(x, mean_test_metrics[:,1], yerr = std_test_metrics[:,1], color= 'blue')
axes[0].set_ylabel('Accuracy')

plt.legend(['Train_data', 'Test_data'])
plt.xlabel('Number of Epochs')

plt.show()
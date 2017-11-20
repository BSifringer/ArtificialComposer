# LSTM for Reber Grammar model loader and plotter
import numpy as np
import matplotlib.pyplot as plt
import shelve
from pandas import read_csv
import math
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from neupy.datasets import make_reber


print('Loading Saved Variables and model')
X = np.load('Xdata.npy')
y = np.load('ydata.npy')


batch_size=len(X[0]) 
n_batch = 50


model = load_model('Cerg_model.h5')


print('Making testdataset')
testdataset = make_reber(batch_size*n_batch)
testdataset = 'O'.join(testdataset)
chars = sorted(list(set(testdataset)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


batches = []
next_char = []
step = 1
for i in range(0, len(testdataset) - batch_size, step):
    batches.append(testdataset[i: i + batch_size])
    next_char.append(testdataset[i + batch_size])
print('test data nb sequences:', len(batches))


print('test data Vectorization...')
Xtest = np.zeros((len(batches), batch_size, len(chars)), dtype=np.bool)
ytest = np.zeros((len(batches), len(chars)), dtype=np.bool)
for j, batch in enumerate(batches):
    for i, data in enumerate(batch):
        Xtest[j, i, char_indices[data]] = 1
    ytest[j, char_indices[next_char[j]]] = 1



print('predicting on trained data and new data')
trainPredict = model.predict(X, batch_size=batch_size)
#model.reset_states() #?? need this?
testPredict = model.predict(Xtest, batch_size=batch_size)

print(trainPredict)
#Need to change/fix this: 


trainScore = math.sqrt(mean_squared_error(y, trainPredict))

testScore = math.sqrt(mean_squared_error(ytest, testPredict))


print('Train Score: %.2f RMSE' % (trainScore))
print('Test Score: %.2f RMSE' % (testScore))

ysequence=np.zeros(len(y),dtype=int)
trainPredictsequence=np.zeros(len(y),dtype=int)


#use argmax or sample to take out chosen indeces
for i in range(len(y)):
	ysequence[i]=np.argmax(y[i])
	trainPredictsequence[i]=np.argmax(trainPredict[i])

plt.hist(np.clip(abs(ysequence-trainPredictsequence),0,1),bins='auto')
plt.show()


ytestsequence=np.zeros(len(ytest),dtype=int)
testPredictsequence=np.zeros(len(ytest),dtype=int)


#use argmax or sample to take out chosen indeces
for i in range(len(ytest)):
	ytestsequence[i]=np.argmax(ytest[i])
	testPredictsequence[i]=np.argmax(testPredict[i])

plt.hist(np.clip(abs(ytestsequence-testPredictsequence),0,1),bins='auto')
plt.show()

#Could be useful: picking out from probabilities instead of argmax
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    print(preds)
    probas = np.random.multinomial(1, preds, 1)
    print(probas)
    return np.argmax(probas)


## invert predictions
#trainPredict = scaler.inverse_transform(trainPredict)
#trainY = scaler.inverse_transform([y])
#testPredict = scaler.inverse_transform(testPredict)
#testY = scaler.inverse_transform([ytest])
## calculate root mean squared error
#trainScore = math.sqrt(mean_squared_error(y, trainPredict))
#print('Train Score: %.2f RMSE' % (trainScore))
#testScore = math.sqrt(mean_squared_error(ytest, testPredict))
#print('Test Score: %.2f RMSE' % (testScore))
## shift train predictions for plotting
#trainPredictPlot = np.empty_like(dataset)
#trainPredictPlot[:, :] = np.nan
#trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
## shift test predictions for plotting
#testPredictPlot = np.empty_like(dataset)
#testPredictPlot[:, :] = np.nan
#testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
## plot baseline and predictions
##plt.plot(scaler.inverse_transform(dataset))
#plt.plot(trainPredictPlot)
#plt.plot(testPredictPlot)
#plt.show()

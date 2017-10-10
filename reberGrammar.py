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


batch_size = 20;
n_batch=50; #redefine this. IT is to make 1000 reber words

iteration=50 #learning


dataset = make_reber(batch_size*n_batch)
dataset = 'O'.join(dataset)
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


#No need:
# normalize the dataset
##scaler = MinMaxScaler(feature_range=(0, 1))
##dataset = scaler.fit_transform(dataset)

#Create Test data later: 
# split into train and test sets
#train_size = int(len(dataset) * 0.67)
#test_size = len(dataset) - train_size
#train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# Already done this with batch input: 
# reshape into X=t and Y=t+1
#look_back = 10
#trainX, trainY = create_dataset(train, look_back)
#testX, testY = create_dataset(test, look_back)

#What does this do...? Matrix form? Already done this .... the tutorial is pretty bad
# reshape input to be [samples, time steps, features]
#trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
#testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))


# create and fit the LSTM network
#batch_size = 1
print('Building the Model layers')
model = Sequential()
model.add(LSTM(80, input_shape=(batch_size,len(chars))))
model.add(Dropout(0.2))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

print('Starting to learn:')
for i in range(iteration):
	print('{} out of {}'.format(i,iteration))
	model.fit(X, y, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
	model.reset_states() # What's this? 


#save the model:
model.save('Cerg_model.h5')
np.save('Xdata.npy',X)
np.save('ydata.npy',y)


## in reber Plotter:
#testdataset = make_reber(batch_size*n_batch)
#testdataset = 'O'.join(testdataset)
#chars = sorted(list(set(testdataset)))
#char_indices = dict((c, i) for i, c in enumerate(chars))#

#batches = []
#next_char = []
#step = 1
#for i in range(0, len(testdataset) - batch_size, step):
#    batches.append(testdataset[i: i + batch_size])
#    next_char.append(testdataset[i + batch_size])
#print('test data nb sequences:', len(batches))#
#

#print('test data Vectorization...')
#Xtest = np.zeros((len(batches), batch_size, len(chars)), dtype=np.bool)
#ytest = np.zeros((len(batches), len(chars)), dtype=np.bool)
#for j, batch in enumerate(batches):
#    for i, data in enumerate(batch):
#        Xtest[j, i, char_indices[data]] = 1
#    ytest[j, char_indices[next_char[j]]] = 1#

## make predictions
#trainPredict = model.predict(X, batch_size=batch_size)
#model.reset_states()
#testPredict = model.predict(Xtest, batch_size=batch_size)


##Need to change this: #
#

## invert predictions
##trainPredict = scaler.inverse_transform(trainPredict)
##trainY = scaler.inverse_transform([y])
##testPredict = scaler.inverse_transform(testPredict)
##testY = scaler.inverse_transform([ytest])
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

# LSTM for Embedded Continuous Reber Grammar 
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
# We will not padd, we will truncate. 
# Make embedded strings size 100T, T = 50 letters
# Make about 100*batch_size of these and batch size in model fit about 16
# look for to categorical from np.utils
# use roll to have a Y which is a shifted X
# (do this before truncating is a great strategy => y won't all finish with B's)
# Check fit models that are stateful, must forget between batches
#		must not forget between steps

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
n_step=1 #redefine this. IT is to make 1000 reber words

batch_size = 16

iteration=50 #learning


dataset = make_reber(n_step*step_size*1000)
dataset = ''.join(dataset)
chars = sorted(list(set(dataset)))
char_indices = dict((c, i) for i, c in enumerate(chars))
print('DataSet concatenated lenght:', len(dataset))

batches = []
next_char = []

for i in range(0, len(dataset) - step_size*n_step, step_size*n_step):
    batches.append(dataset[i: i + step_size*n_step])
print('nb strings of size {}:'.format(step_size*n_step), len(batches))


print('Vectorization...')
X = np.zeros((len(batches), step_size*n_step, len(chars)), dtype=np.bool)
for j, batch in enumerate(batches):
    for i, data in enumerate(batch):
        X[j, i, char_indices[data]] = 1

X=X[:len(X)-(len(X)%batch_size)]
Y=np.roll(X,-1,axis=0)

# create and fit the LSTM network
print('Building the Model layers')
model = Sequential()
model.add(LSTM(80, batch_input_shape=(batch_size,step_size,len(chars)),stateful=True))
model.add(Dropout(0.2))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

print('Starting to learn:')
for i in range(iteration):
	print('{} out of {}'.format(i,iteration))
	#check parameters, shuffle True? Stateful False? 
	model.fit(X, Y, epochs=1, batch_size=batch_size, verbose=2, shuffle=True)
	model.reset_states() # What's this? 


#save the model:
model.save('embedCerg_model.h5')
np.save('embedXdata.npy',X)
np.save('embedydata.npy',Y)


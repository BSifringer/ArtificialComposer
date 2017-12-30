from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import random
#import matplotlib.pyplot as plt
import shelve
#from pandas import read_csv, get_dummies
from keras.models import Sequential, Model
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.layers import Input, Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, Add, Reshape, Subtract, Multiply
from keras.optimizers import RMSprop, Adam
from keras.utils import np_utils, plot_model
#import matplotlib.pyplot as plt


#input1 = Input([1])
#input2 = Input([1])
#add = Add()([input1, input2])
#multiply = Multiply(name = 'multiply')([add,input2])
#subtract = Subtract()([multiply,input2])
#model1 = Model([input1, input2], subtract)

#
#new_input = model1.input
#new_output = model1.get_layer(name='multiply').output
#model2 = Model(new_input, new_output)	
#
#print(model2.predict([np.array([5]),np.array([2])]))

model = load_model("backward_model.h5")

model = Model(model.layers[0].input,model.layers[5].output)
model.layers[0].name = 'Y'
model.layers[1].name = 'Filter'
model.layers[2].name = '1st Layer'
model.layers[3].name = '2nd Layer'
model.layers[4].name = 'Regularizer'
model.layers[5].name = 'Dense Layer'
model.layers[5].layer.name = ''
model.layers[6].name = 'X'

input1 = model.layers[0].input
output2 = model.layers[6].output
 
model2 = Model(input1,output2)

plot_model(model2, to_file = 'backward_model.png' )
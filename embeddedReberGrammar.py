
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

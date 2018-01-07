import numpy as np
import shelve
#from pandas import read_csv, get_dummies
import math
import random
#import tensorflow as tf

from keras.utils import np_utils

from utils import *
#import matplotlib.pyplot as plt


def get_model_input(song_sizes = 300, n_step = 5, percentage = 0.1, add_time = False, reverse = False, Minimum_batches = 10):
	
	data_percentage = min(percentage,1)

	max_song_size = song_sizes + 1 #(based on histogram) (extra 1 since we roll Y and delete last column)
	step_size = int(max_song_size / n_step) # 100 ~ upper limit
	test_percentage = 0.1 # for number of test batches


	print("loading songs")

	data, dictionaries = load("../BobSturm.pkl")

	pitch_indices = dict((p,i) for i, p in enumerate(dictionaries["pitchseqs"]))
	pitch_data = np.array(data["pitchseqs"][:int(data_percentage*len(data["pitchseqs"]))])

	song_sizes = np.array([len(i) for i in pitch_data]) #do a plt.hist(song_sizes) plt.show()
	remove_idx = np.where(song_sizes>max_song_size)

	song_sizes = np.delete(song_sizes, remove_idx)
	pitch_data = np.delete(pitch_data, remove_idx)

	batch_possible = [ k for k in range(1,int(pitch_data.size/Minimum_batches)) if pitch_data.size%k == 0]

	batch_size = int(batch_possible[-1])
	n_batch = int(pitch_data.size/batch_size)

	if n_batch > 10*Minimum_batches: 
		batch_size = int(pitch_data.size/(10*Minimum_batches))
		n_batch = int(pitch_data.size/batch_size) + 1

	n_test_batch = max(int(n_batch*test_percentage),1)

	train_idxes = np.array(range(0,(n_batch-n_test_batch)*batch_size))
	test_idxes = np.array(range((n_batch-n_test_batch)*batch_size, n_batch*batch_size))

	batches = []
	next_pitch = []

	print('Vectorization...')

	X = -1*np.ones((n_batch*batch_size, step_size*n_step+1, len(pitch_indices)), dtype=np.int) #or float64
	for j, pitches in enumerate(pitch_data):
		categorized = np_utils.to_categorical([pitch_indices[p] for p in pitches], num_classes = 47)
		##if truncating and masking: 
		#X[j,:min(len(pitches,),step_size*n_step)] =  categorized[:min(len(pitches),step_size*n_step)]
		# if masking only:
		X[j,:len(pitches,)] =  categorized
	
	if reverse:
		print('Reversing data...')
		X = np.flip(X,axis=1)


	Y = np.roll(X,-1,axis=1)
	X = np.delete(X,-1,axis=1)
	Y = np.delete(Y,-1,axis=1)
	

	# Adding next pitch duration as input:
	time_dim = 0
	if add_time:
		print('Adding Time...')
		time_indices = dict((p,i) for i, p in enumerate(dictionaries["Tseqs"]))
		time_dim = len(time_indices)
		time_data =  np.array(data["Tseqs"][:int(data_percentage*len(data["Tseqs"]))])
		time_data = np.delete(time_data, remove_idx)
		
		X2 = -1*np.ones((n_batch*batch_size, step_size*n_step+1, time_dim), dtype=np.int) #or float64
		for j, times in enumerate(time_data):
			categorized = np_utils.to_categorical([time_indices[t] for t in times], num_classes = time_dim )
			##if truncating and masking: 
			#X[j,:min(len(pitches,),step_size*n_step)] =  categorized[:min(len(pitches),step_size*n_step)]
			# if masking only:
			X2[j,:len(times,)] = categorized
		if reverse:
			X2 = np.flip(X2, axis=1)
		X2 = np.roll(X2,-1,axis=1)
		X2 = np.delete(X2,-1,axis=1)
		X = np.concatenate((X,X2), axis = 2)


	return X, Y, n_batch, n_test_batch

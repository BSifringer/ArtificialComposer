import numpy as np

# load songs
# songs = np.load('MonophonicMusicModel\\data\\BobSturm.npy')
songs = np.load('data\\BobSturm.npy')
pitches = sorted(set.union(*[set(s.pitch) for s in songs]))
durations = sorted(set.union(*[set(s.dt) for s in songs]))

# model specifications
step_size = 60                          # size of step within song (number of notes trained per batch and song)
n_pitches = len(pitches)                # number of different pitches
n_durations = len(durations)            # number of different note durations
dropout_rate = 0.1                      # dropout rate
batch_size = 50                         # size of batch (number of songs trained on in parallel)
model_name = 'model4'                   # model name (file name of stored model)
# model1: first tests; model2: x-model; model3: x-model + end layer; model4: model2_stateful + end pitch

# dictionaries tp transform pitches / durations into integer indices
pitch2index = {p: i for i, p in enumerate(pitches)}
index2pitch = {i: p for i, p in enumerate(pitches)}
duration2index = {d: i for i, d in enumerate(durations)}
index2duration = {i: d for i, d in enumerate(durations)}

# training specifications
hyper_parameter = 'test5'               # hyper parameter for tensorboard
max_song_size = 300                     # maximum song size taken from data
n_step = int(max_song_size/step_size)   # number of training steps per song
train_percentage = 0.8                  # percentage of data to train on
n_batch = 910  # 916                    # number of batches per epoch (there are 45849 songs)
n_epoch = 15                            # number of epochs

# generation specifications
duration_entropy_power = 1
pitch_entropy_power = 1
end_entropy_power = 1
max_generated_song_size = 300
number_of_songs = 50
generation_epoch = 34
generation_model_name = 'model4'
# TODO:
# pitch from rhythm: generate data from given rhythms -> X-model generates both now ;) (v)
# shortcut-connections: see bachprop (v)
# learn continuously (masking layer does it) (v)
# generate Data !!!
# test interval_n_gram
# end tag (evt)
# stats on song length
# shift pitches randomly and shuffle songs during training / after each epoch (v)
# test-data to validate against over-fitting (v)
# name layers in keras
# n-gram vs entropy

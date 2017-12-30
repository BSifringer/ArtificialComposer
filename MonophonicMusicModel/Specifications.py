import MusicUtility.LoadStore

# load songs
# songs = MusicUtility.LoadStore.load_pkl('MonophonicMusicModel\\data\\BobSturm.pkl')
songs = MusicUtility.LoadStore.load_pkl('data\\BobSturm.pkl')
pitches = sorted(set.union(*[set(s.pitch) for s in songs]))
durations = sorted(set.union(*[set(s.dt) for s in songs]))

# model specifications
step_size = 60                          # size of step within song (number of notes trained per batch and song)
n_pitches = len(pitches)                # number of different pitches
n_durations = len(durations)            # number of different note durations
dropout_rate = 0.1                      # dropout rate
model_name = 'model2'                   # model name (file name of stored model)

# dictionaries tp transform pitches / durations into integer indices
pitch2index = {p: i for i, p in enumerate(pitches)}
index2pitch = {i: p for i, p in enumerate(pitches)}
duration2index = {d: i for i, d in enumerate(durations)}
index2duration = {i: d for i, d in enumerate(durations)}

# training specifications
hyper_parameter = 'test2'               # hyper parameter for tensorboard
max_song_size = 300                     # maximum song size taken from data
n_step = int(max_song_size/step_size)   # number of training steps per song
train_percentage = 0.8                  # percentage of data to train on
batch_size = 50                         # size of batch (number of songs trained on in parallel)
n_batch = 910  # 916                    # number of batches per epoch (there are 45849 songs)
n_epoch = 10                            # number of epochs


# TODO:
# pitch from rhythm: generate data from given rhythms -> X-model generates both now ;) (v)
# shortcut-connections: see bachprop (v)
# learn continuously (masking layer does it) (v)
# generate Data !!!
# test interval_n_gram
# end tag (evt)
# stats on song length
# shift pitches randomly and shuffle songs during training / after each epoch
# test-data to validate against over-fitting
# name layers in keras

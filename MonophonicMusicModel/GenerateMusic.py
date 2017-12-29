import numpy as np
from keras.models import load_model
from MonophonicMusicModel.Specifications import *
from MusicUtility.Song import Song
import MusicUtility.LoadStore as ls

model = load_model('models\\'+model_name+'_epoch_'+str(24)+'.h5')

generated_t = [12]
generated_pitch = [pitch2index[60]]


def probabilistic_arg_max(p):
    """ returns index of p-element with a probability according to its value """
    p_sum = np.cumsum(p)
    random = p_sum[-1]*np.random.rand()
    return np.sum(p_sum < random)


model.reset_states()

start_duration = np.zeros([1, step_size, n_durations])
start_pitch = np.zeros([1, step_size, n_pitches])
start_duration[0, 0, 12] = 1
start_pitch[0, 0, pitch2index[60]] = 1
y = model.predict([start_duration, start_pitch])
generated_t.append(probabilistic_arg_max(y[0][0, 0]))
generated_pitch.append(probabilistic_arg_max(y[1][0, 0]))
for i in range(1, step_size):
    start_duration[0, i, generated_t[-1]] = 1
    start_pitch[0, i, generated_pitch[-1]] = 1
    y = model.predict_on_batch([start_duration, start_pitch])
    generated_t.append(probabilistic_arg_max(y[0][0, i]))
    generated_pitch.append(probabilistic_arg_max(y[1][0, i]))
    model.reset_states()

generated_t = [index2duration[t] for t in generated_t]
generated_pitch = [index2pitch[p] for p in generated_pitch]
generated_song = Song(generated_t, generated_pitch, generated_t)
ls.store_midi(generated_song, 'generated music\\epoch 24\\', 'test4')

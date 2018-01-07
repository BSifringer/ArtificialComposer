import numpy as np
from keras.models import load_model
from MonophonicMusicModel.Specifications import *
from MusicUtility.Song import Song
import MusicUtility.LoadStore as ls

model = load_model('models\\'+model_name+'_epoch_'+str(24)+'.h5')


def probabilistic_arg_max(p):
    """ returns index of p-element with a probability according to its value """
    p_sum = np.cumsum(p)
    random = p_sum[-1]*np.random.rand()
    return np.sum(p_sum < random)


model.reset_states()

generated_t = [duration2index[1]]
generated_pitch = [pitch2index[60]]

start_duration = np.zeros([1, step_size, n_durations])
start_pitch = np.zeros([1, step_size, n_pitches])

stopped = False

for i in range(1, step_size):
    start_duration[0, i, generated_t[-1]] = 1
    start_pitch[0, i, generated_pitch[-1]] = 1
    y = model.predict_on_batch([start_duration, start_pitch])
    print('step {}: stop probability: {}'.format(i, y[2][0, i][0]))
    if probabilistic_arg_max(y[2][0, i]) == 0:
        stopped = True
        print("music stopped")
        break
    generated_t.append(probabilistic_arg_max(np.power(y[0][0, i], duration_entropy_power)))
    generated_pitch.append(probabilistic_arg_max(np.power(y[1][0, i], pitch_entropy_power)))
    model.reset_states()

if not stopped:
    for i in range(step_size, max_generated_song_size):
        start_duration = np.roll(start_duration, -1, axis=1)
        start_pitch = np.roll(start_pitch, -1, axis=1)
        start_duration[0, step_size-1, generated_t[-1]] = 1
        start_pitch[0, step_size-1, generated_pitch[-1]] = 1
        y = model.predict_on_batch([start_duration, start_pitch])
        print('step {}: stop probability: {}'.format(i, y[2][0, step_size-1][0]))
        if probabilistic_arg_max(y[2][0, step_size-1]) == 0:
            stopped = True
            print("music stopped")
            break
        generated_t.append(probabilistic_arg_max(np.power(y[0][0, step_size-1], duration_entropy_power)))
        generated_pitch.append(probabilistic_arg_max(np.power(y[1][0, step_size-1], pitch_entropy_power)))
        model.reset_states()

generated_t = [index2duration[t] for t in generated_t]
generated_pitch = [index2pitch[p] for p in generated_pitch]
generated_song = Song(generated_t, generated_pitch, generated_t)
ls.store_midi(generated_song, 'generated music\\model3\\epoch 34\\', 'test1', 'normal')

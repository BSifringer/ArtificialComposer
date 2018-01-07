import numpy as np
from keras.models import load_model
from MonophonicMusicModel.Specifications import *
from MusicUtility.Song import Song
import MusicUtility.LoadStore as ls

# model = load_model('MonophonicMusicModel\\models\\'+model_name+'_epoch_'+str(19)+'.h5')
model = load_model('models\\'+generation_model_name+'_epoch_'+str(generation_epoch)+'.h5')
generation_path = 'generated music\\'+generation_model_name+'\\epoch '+str(generation_epoch)+'\\'


def probabilistic_arg_max(p):
    """ returns index of p-element with a probability according to its value """
    p_sum = np.cumsum(p)
    random = p_sum[-1]*np.random.rand()
    return np.sum(p_sum < random)


generated_songs = []
number_of_batches = int(number_of_songs/batch_size)
number_of_songs = number_of_batches*batch_size

for batch_number in range(number_of_batches):
    print('generating batch nr. {} of {}'.format(batch_number+1, number_of_batches))
    model.reset_states()

    # pick random songs and take their start pitches / length to make proper statistics
    # add random offset between -6 / +6 to every start pitch
    start_indices = np.random.randint(low=0, high=len(songs), size=batch_size)
    generated_t = [[duration2index[songs[i].t[0]]] for i in start_indices]
    generated_pitch = [[pitch2index[songs[i].pitch[0]]+int(np.random.randint(
                            max(-pitch2index[songs[i].pitch[0]], -6),
                            min(n_pitches-1-pitch2index[songs[i].pitch[0]], 6), 1))] for i in start_indices]

    # generate new values up to max_generated_song_size
    for i in range(max_generated_song_size):
        feed_duration = np.zeros([batch_size, step_size, n_durations])
        feed_pitch = np.zeros([batch_size, step_size, n_pitches])
        for j in range(batch_size):
            if generated_pitch[j][-1] != n_pitches:
                feed_duration[j, 0, generated_t[j][-1]] = 1
                feed_pitch[j, 0, generated_pitch[j][-1]] = 1
        y = model.predict_on_batch([feed_duration, feed_pitch])

        for j in range(batch_size):
            if generated_pitch[j][-1] != n_pitches:
                generated_t[j].append(probabilistic_arg_max(np.power(y[0][j, 0], duration_entropy_power)))
                generated_pitch[j].append(probabilistic_arg_max(np.power(y[1][j, 0], pitch_entropy_power)))
        print('generated {} notes out of {}'.format(i+1, max_generated_song_size))

    # translate model results into midi codecs
    for i in range(batch_size):
        generated_t[i] = [index2duration[t] for t in generated_t[i][:-1]]
        generated_pitch[i] = [index2pitch[p] for p in generated_pitch[i][:-1]]
        generated_songs.append(Song(generated_t[i], generated_pitch[i], generated_t[i]))

# save generated songs

# ... as npy file
np.save(generation_path+'{}_batch_generated_songs_e_{}_p_{}_d_{}.npy'.format(
    number_of_songs, end_entropy_power, pitch_entropy_power, duration_entropy_power), generated_songs)

# ... and as midi
for song_number in range(number_of_songs):
    ls.store_midi(generated_songs[song_number],
                  generation_path,
                  'batch_generated_song_{}'.format(song_number),
                  '_e_{}_p_{}_d_{}'.format(end_entropy_power, pitch_entropy_power, duration_entropy_power))

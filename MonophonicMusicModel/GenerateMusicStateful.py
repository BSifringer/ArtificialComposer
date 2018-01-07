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
for song_number in range(number_of_songs):
    print('generating song nr. {}'.format(song_number))
    model.reset_states()

    # TODO: random start pitch / duration (maybe choose random song), generate 50 songs at a time

    # H-A input
    generated_t = [duration2index[1], duration2index[1]]
    generated_pitch = [pitch2index[71], pitch2index[69]]

    # 3/4 input
    # generated_t = [duration2index[1], duration2index[1], duration2index[1],
    #                duration2index[1], duration2index[1], duration2index[1]]
    # generated_pitch = [pitch2index[55], pitch2index[60], pitch2index[60],
    #                    pitch2index[55], pitch2index[60], pitch2index[60]]

    # 4/4 input
    # generated_t = [duration2index[1], duration2index[1], duration2index[1], duration2index[1],
    #                duration2index[1], duration2index[1], duration2index[1], duration2index[1]]
    # generated_pitch = [pitch2index[55], pitch2index[60], pitch2index[62], pitch2index[60],
    #                    pitch2index[55], pitch2index[60], pitch2index[64], pitch2index[60]]

    # feed first values of generated_t / generated_pitch into model
    for i in range(0, len(generated_pitch)-1):
        feed_duration = np.zeros([batch_size, step_size, n_durations])
        feed_pitch = np.zeros([batch_size, step_size, n_pitches])
        feed_duration[0, 0, generated_t[i]] = 1
        feed_pitch[0, 0, generated_pitch[i]] = 1
        y = model.predict_on_batch([feed_duration, feed_pitch])
        print('generated {} notes out of {}'.format(i+1, max_generated_song_size))

    # generate new values up to max_generated_song_size
    for i in range(len(generated_pitch)-1, max_generated_song_size):
        feed_duration = np.zeros([batch_size, step_size, n_durations])
        feed_pitch = np.zeros([batch_size, step_size, n_pitches])
        feed_duration[0, 0, generated_t[-1]] = 1
        feed_pitch[0, 0, generated_pitch[-1]] = 1
        y = model.predict_on_batch([feed_duration, feed_pitch])
        end = np.power(y[1][0, 0][n_pitches], end_entropy_power)
        print('end probability: {}'.format(end), end=' ')
        if end > np.random.rand():
            break
        generated_t.append(probabilistic_arg_max(np.power(y[0][0, 0], duration_entropy_power)))
        generated_pitch.append(probabilistic_arg_max(np.power(y[1][0, 0][:n_pitches], pitch_entropy_power)))
        print('generated {} notes out of {}'.format(i+1, max_generated_song_size))

    # translate model results into midi codecs
    generated_t = [index2duration[t] for t in generated_t]
    generated_pitch = [index2pitch[p] for p in generated_pitch]
    generated_songs.append(Song(generated_t, generated_pitch, generated_t))

    # save generated song
    ls.store_midi(generated_songs[song_number],
                  generation_path,
                  'song_{}'.format(song_number),
                  '_e_{}_p_{}_d_{}'.format(end_entropy_power, pitch_entropy_power, duration_entropy_power))

np.save(generation_path+'{}_generated_songs_e_{}_p_{}_d_{}.npy'.format(
    number_of_songs, end_entropy_power, pitch_entropy_power, duration_entropy_power), generated_songs)

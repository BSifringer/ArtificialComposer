import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.callbacks import TensorBoard
from MonophonicMusicModel.Specifications import *

model = load_model('models\\'+model_name+'.h5')


def generate_input(song_indices, note_indices, pitch_shifts=np.zeros(batch_size)):
    """ generate network input for given song indices and note indices (that should be of length batch_size)
        pitch_shifts are shifting all the songs by the given amount and useful to augment the training data"""
    duration_input = np.zeros([len(song_indices), len(note_indices), n_durations])
    pitch_input = np.zeros([len(song_indices), len(note_indices), n_pitches])

    for i in range(len(song_indices)):
        for j in [tmp for tmp in range(len(note_indices)) if note_indices[tmp] < len(songs[song_indices[i]].pitch)]:
            if songs[song_indices[i]].pitch[note_indices[j]]+pitch_shifts[i] in pitches:
                pitch_input[i, j, pitch2index[songs[song_indices[i]].pitch[note_indices[j]]+pitch_shifts[i]]] = 1
            if songs[song_indices[i]].t[note_indices[j]] in durations:
                duration_input[i, j, duration2index[songs[song_indices[i]].t[note_indices[j]]]] = 1
    return [duration_input, pitch_input]


def generate_output(song_indices, note_indices, pitch_shifts=np.zeros(batch_size)):
    """ convert integer song to first layer input of shape len(songs) x songs_length x (pitches + beats + 1)_length
        the size of last input dimension is increased by 1 in order to have an ending
        use this method only directly before feeding data into the training-procedure in order to save storage """
    duration_output = np.zeros([len(song_indices), step_size, n_durations])
    pitch_output = np.zeros([len(song_indices), step_size, n_pitches])

    for i in range(len(song_indices)):
        for j in [tmp for tmp in range(len(note_indices)) if note_indices[tmp]+1 < len(songs[song_indices[i]].pitch)]:
            if songs[song_indices[i]].pitch[note_indices[j]+1]+pitch_shifts[i] in pitches:
                pitch_output[i, j, pitch2index[songs[song_indices[i]].pitch[note_indices[j]+1]+pitch_shifts[i]]] = 1
            if songs[song_indices[i]].t[note_indices[j]+1] in durations:
                duration_output[i, j, duration2index[songs[song_indices[i]].t[note_indices[j]+1]]] = 1
    return [duration_output, pitch_output]


# create a Tensorboard callback -> used to supervise training progress
# >tensorboard.exe --logdir=C:/Users/NiWa/PycharmProjects/ArtificialComposer/MonophonicMusicModel/tensorboard_logs

callback = TensorBoard('tensorboard_logs/'+hyper_parameter)
callback.set_model(model)


def write_log(names, logs, batch_no):
    """ write a log into a tensorflow callback """
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


train_indices = np.arange(0, int(train_percentage*len(songs)))
n_train_batches = int(len(train_indices)/batch_size)
test_indices = np.arange(int(train_percentage*len(songs)), len(songs))
n_test_batches = int(len(test_indices)/batch_size)

for e in range(n_epoch):
    print('Epoch {} / {}'.format(e, n_epoch))
    np.random.shuffle(train_indices)
    for i in range(n_train_batches):
        print('Train Batch {} / {}'.format(i, n_train_batches))
        song_indices = train_indices[i*batch_size:(i+1)*batch_size]
        pitch_shifts = np.random.randint(-6, 6, batch_size)
        for j in range(n_step):
            note_indices = np.arange(j*step_size, (j+1)*step_size)
            metrics = model.train_on_batch(x=generate_input(song_indices, note_indices, pitch_shifts),
                                           y=generate_output(song_indices, note_indices, pitch_shifts))
            write_log(['train_'+n for n in model.metrics_names], metrics, e*n_step*n_train_batches+i*n_step+j)
        model.reset_states()

    model.save('models\\'+model_name+'_epoch_'+str(e)+'.h5')

    np.random.shuffle(test_indices)
    for i in range(n_test_batches):
        print('Test Batch {} / {}'.format(i, n_test_batches))
        song_indices = test_indices[i*batch_size:(i+1)*batch_size]
        pitch_shifts = np.random.randint(-6, 6, batch_size)
        for j in range(n_step):
            note_indices = np.arange(j*step_size, (j+1)*step_size)
            metrics = model.test_on_batch(x=generate_input(song_indices, note_indices, pitch_shifts),
                                          y=generate_output(song_indices, note_indices, pitch_shifts))
            write_log(['test_'+n for n in model.metrics_names], metrics, e*n_step*n_test_batches+i*n_step+j)
        model.reset_states()

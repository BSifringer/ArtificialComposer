from keras.models import load_model
from keras.utils import np_utils
import matplotlib.pyplot as plot
import random
from utils import *
# -> for Brian: change this back to utils ;)
#from  ..MusicUtility import Statistics
#from ..MusicUtility import Song

if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir
    
    path.append(dir(path[0]))
    __package__ = "monoPhonic"

#import MusicUtility.Statistics as Stat
#import MusicUtility.Song as Song
# Do we predict a single batch or a full song?
# !! Input shape has shape of batch
# For rythm, take a random song? 
# Generate until random song is empty ? => Multiple batches? 
# Idea: never reset states, use predict on single note and rythm and save in vector => no need to work with changing batches

plotting = False
write_midi = True
#print_stats = True

print('Loading model')
#model = load_model('monoPhonic_model2.h5')
if 'model' not in vars():
#   model = load_model('monoPhonic_model2.h5')
    model = load_model('forward_model_single.h5')
else:
    model.reset_states()

pitch_indices = {i + 48: i for i in range(47)}

indices_pitch = dict((i, p) for i, p in enumerate(pitch_indices))

input_shape = (model.get_layer(index=0)).input_shape

xPredictBatch = -1 * np.ones(input_shape)

xPredictBatch[0, 0] = np.zeros(len(xPredictBatch[0, 0]));

added_time = len(xPredictBatch[0, 0]) > 50

predict_size = input_shape[1]

if added_time:
    print('Vectorizing rhythm from random song')
    data, dictionaries = load("BobSturm.pkl")
    time_indices = dict((p, i) for i, p in enumerate(dictionaries["Tseqs"]))
    time_dim = len(time_indices)
    time_data = np.array(data["Tseqs"])
    random_rhythm = random.choice(time_data)
    predict_size = len(random_rhythm)
    # xPredictBatch[0,:min(len(random_rythm),len(input_shape[0,:,0])), -time_dim:] = np_utils.to_categorical([time_indices[t] for t in random_rythm], num_classes = time_dim)

# add 1 in pitch len, and 1 in rest

prediction = np.zeros(predict_size)
statistic = np.zeros([2, predict_size, len(pitch_indices)])
start = random.choice(range(len(pitch_indices)))
xPredictBatch[0, 0, start] = 1

print('Predicting')
for i in range(predict_size):
    if added_time:
        xPredictBatch[0, 0, -time_dim:] = np_utils.to_categorical(time_indices[random_rhythm[i]], num_classes=time_dim)
    P = model.predict(xPredictBatch, batch_size=input_shape[0])
    if len(P) == 2:
        P = P[0]
    # choice = np.argmax(P[0,i]) #Temporary probability
    # choice = np.argmax(P[0,0]) #Temporary probability
    choice = np.random.choice(range(len(pitch_indices)), p=P[0, 0])  # take out index value (Range) from P probabilities

    statistic[1, i] = xPredictBatch[0, 0, :len(pitch_indices)]

    xPredictBatch[0, 0] = np.zeros(input_shape[2])
    xPredictBatch[0, 0, choice] = 1
    prediction[i] = indices_pitch[choice]

    statistic[0, i] = P[0, 0]
    statistic[1, i] = xPredictBatch[0, 0, :len(pitch_indices)]

#
if plotting:
    imgplot = plot.imshow(statistic.transpose(0, 2, 1)[0])
    plot.yticks(np.arange(statistic[0].shape[1]), list(indices_pitch.values()))
    plot.xticks(np.arange(statistic[0].shape[0]), [indices_pitch[np.argmax(i)] for i in statistic[1, :, :]])
    # plot.title('Generated Reber Word sampled with argmax')
    # plot.colorbar()
    plot.show()

# To be more correct, random_rythm should be rolled -1 upon training
# The output here should also be the first random note + predictions
prediction = np.array(prediction, dtype=int)
if write_midi:
    writeMIDI(random_rhythm, prediction, label="test_forward_single")

# do some statistics

#if print_stats:
#    song = Song.Song(random_rhythm, prediction, random_rhythm)
#
#    plot.figure(1)
#    pitch_histogram = Stat.pitch_histogram([song], Stat.pitches)
#    plot.plot(pitch_histogram.keys(), pitch_histogram.values())
#    plot.xlabel("key")
#    plot.ylabel("frequency")
#    plot.draw()
#
#    plot.figure(2)
#    following_pitches_histogram = Stat.following_pitches_histogram([song], Stat.pitches)
#    plot.imshow([[following_pitches_histogram[(i, j)] for i in Stat.pitches] for j in Stat.pitches], origin='lower')
#    plot.xlabel("key 1")
#    plot.ylabel("key 2")
#    plot.draw()
#
#    plot.figure(3)
#    interval_histogram = Stat.interval_histogram([song], Stat.pitches)
#    plot.plot(interval_histogram.keys(), interval_histogram.values())
#    plot.xlabel("interval")
#    plot.ylabel("frequency")
#    plot.show()

import MusicUtility.LoadStore as ls
import MusicUtility.Statistics as stat
import matplotlib.pyplot as plot
from MonophonicMusicModel.Specifications import *
import numpy as np

# look up: PCA , t-SNE -> apply to features we want to compare
# n-th order marcov chains, difference measure to predictions of model

generated_songs = np.load('generated music\\model4\\epoch 29\\1000_batch_generated_songs_e_1_p_1_d_1.npy')


def smooth_histogram(values, n):
    values = np.asarray(values)
    for i in range(n):
        values = 0.5*(values[1:]+values[:-1])
    return values


def plot_histogram(songs, smoothing, note_length=None, uniform_smooth=False):
    pitch_histogram = stat.pitch_histogram(songs, pitches, note_length)
    keys = pitch_histogram.keys()
    pitch_histogram = smooth_histogram(list(pitch_histogram.values()), smoothing)
    if uniform_smooth is True:
        pitch_histogram = pitch_histogram[:-6]+pitch_histogram[1:-5]+pitch_histogram[2:-4]+pitch_histogram[3:-3] +\
                          pitch_histogram[4:-2]+pitch_histogram[5:-1]+pitch_histogram[6:]
    plot.plot(stat.normalize(pitch_histogram))
    # plot.plot(keys, stat.normalize(pitch_histogram))
    plot.xlabel('key')
    plot.ylabel('probability')
    plot.draw()


def plot_following_pitches(songs):
    following_pitches_histogram = stat.following_pitches_histogram(songs, pitches)
    plot.imshow([[following_pitches_histogram[(i, j)] for i in pitches] for j in pitches], origin='lower')
    plot.xlabel('key 1')
    plot.ylabel('key 2')
    plot.draw()


def plot_intervals(songs, smoothing, note_length=None):
    interval_histogram = stat.interval_histogram(songs, pitches, note_length)
    keys = interval_histogram.keys()
    interval_histogram = smooth_histogram(list(interval_histogram.values()), smoothing)
    plot.plot(keys, stat.normalize(interval_histogram))
    plot.xlabel('interval')
    plot.ylabel('probability')
    plot.draw()


def plot_following_intervals(songs):
    following_pitches_histogram = stat.following_intervals_histogram(songs, pitches)
    plot.imshow([[following_pitches_histogram[(i, j)]
                  for i in range(min(pitches)-max(pitches), max(pitches)-min(pitches)+1)]
                 for j in range(min(pitches)-max(pitches), max(pitches)-min(pitches)+1)], origin='lower',
                extent=[min(pitches)-max(pitches), max(pitches)-min(pitches)+1,
                        min(pitches)-max(pitches), max(pitches)-min(pitches)+1])
    plot.xlabel('1. interval')
    plot.ylabel('2. interval')
    plot.draw()


def plot_n_gram(songs, n):
    n_gram = stat.n_gram(songs, n)
    plot.plot(n_gram.values())
    plot.xlabel('{} - grams'.format(n))
    plot.ylabel('frequency')
    plot.draw()
    print(n_gram)


def plot_interval_n_gram(songs, n):
    interval_n_gram = stat.interval_n_gram(songs, n)
    plot.plot(interval_n_gram.values())
    plot.xlabel('{} - interval grams'.format(n))
    plot.ylabel('frequency')
    plot.draw()


def plot_song_length_histogram(songs, smoothing):
    song_length_histogram = smooth_histogram(list(stat.song_length_histogram(songs, max_song_size).values()), smoothing)
    plot.plot(stat.normalize(song_length_histogram))
    plot.xlabel('song length')
    plot.ylabel('probability')
    plot.draw()


def plot_n_gram_entropy(songs, n):
    n_gram_entropy = stat.n_gram_entropy(songs, n)
    plot.plot(n_gram_entropy.values())
    plot.xlabel('n-gram')
    plot.ylabel('entropy')
    plot.draw()


def plot_interval_n_gram_entropy(songs, n):
    n_gram_entropy = stat.interval_n_gram_entropy(songs, n)
    plot.plot(n_gram_entropy.values())
    plot.xlabel('n-gram')
    plot.ylabel('entropy')
    plot.draw()


#  pitch histograms
# plot.figure(2)
# plot_histogram(songs, 0, uniform_smooth=True)
# plot.title('pitch histogram - database')
# plot.figure(3)
# plot_histogram(generated_songs, 0)
# plot.title('pitch histogram - generated songs')

#  pitch histograms for different note lengths
# plot.figure(4)
# plot_histogram(songs, 5, 2, True)
# plot.hold(True)
# plot_histogram(songs, 5, 1, True)
# plot_histogram(songs, 5, 0.5, True)
# plot_histogram(songs, 5, 0.25, True)
# plot.title('pitch histogram database')
# plot.legend(['2', '1', '0.5', '0.25'])
# plot.figure(5)
# plot_histogram(generated_songs, 5, 2)
# plot.hold(True)
# plot_histogram(generated_songs, 5, 1)
# plot_histogram(generated_songs, 5, 0.5)
# plot_histogram(generated_songs, 5, 0.25)
# plot.title('pitch histogram generated songs')
# plot.legend(['2', '1', '0.5', '0.25'])

#  interval histograms
# plot.figure(3)
# plot_intervals(songs, 0)
# plot.title('interval histogram - database')
# plot.figure(4)
# plot_intervals(generated_songs, 0)
# plot.title('interval histogram - generated songs')

#  interval histograms for different target note lengths
# plot.figure(1)
# plot_intervals(songs, 0, 2)
# plot.hold(True)
# plot_intervals(songs, 0, 1)
# plot_intervals(songs, 0, 0.5)
# plot_intervals(songs, 0, 0.25)
# plot.title('interval histogram - database')
# plot.legend(['2', '1', '0.5', '0.25'])
# plot.figure(2)
# plot_intervals(generated_songs, 0, 2)
# plot.hold(True)
# plot_intervals(generated_songs, 0, 1)
# plot_intervals(generated_songs, 0, 0.5)
# plot_intervals(generated_songs, 0, 0.25)
# plot.title('interval histogram - generated songs')
# plot.legend(['2', '1', '0.5', '0.25'])

#  plot 1-gram interval heat map
plot.figure(1)
plot_following_intervals(songs)
plot.title('2 following intervals - database')
plot.figure(2)
plot_following_intervals(generated_songs)
plot.title('2 following intervals - generated songs')

plot.show()


#  old stuff
# plot_n_gram_entropy(5)
# plot_interval_n_gram_entropy(5)
# plot_histogram()
# plot_following_pitches()
# plot_intervals()
# plot_n_gram(4)  # zipf's law? (1/x)
# plot_interval_n_gram(4)

# plot.figure(1)
# plot_song_length_histogram(songs, 10)
# plot.hold(True)
# plot_song_length_histogram(generated_songs, 10)
# plot.title('song length histogram')

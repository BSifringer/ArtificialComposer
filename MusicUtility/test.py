import MusicUtility.LoadStore as ls
import MusicUtility.Statistics as stat
import matplotlib.pyplot as plot

# look up: PCA , t-SNE -> apply to features we want to compare
# n-th order marcov chains, difference measure to predictions of model

songs = ls.load_pkl("C:\\Users\\NiWa\\PycharmProjects\\ArtificialComposer\\Monophonic\\BobSturm.pkl")
# ls.store_midi(songs[1],path="C:/Users/NiWa/Desktop/",label="test") # store a song


def plot_histogram():
    plot.figure(1)
    pitch_histogram = stat.pitch_histogram(songs, stat.pitches)
    plot.plot(pitch_histogram.keys(), pitch_histogram.values())
    plot.xlabel("key")
    plot.ylabel("frequency")
    plot.draw()


def plot_following_pitches():
    plot.figure(2)
    following_pitches_histogram = stat.following_pitches_histogram(songs, stat.pitches)
    plot.imshow([[following_pitches_histogram[(i, j)] for i in stat.pitches] for j in stat.pitches], origin='lower')
    plot.xlabel("key 1")
    plot.ylabel("key 2")
    plot.draw()


def plot_intervals():
    plot.figure(3)
    interval_histogram = stat.interval_histogram(songs, stat.pitches)
    plot.plot(interval_histogram.keys(), interval_histogram.values())
    plot.xlabel("interval")
    plot.ylabel("frequency")
    plot.draw()


def plot_n_gram(n):
    plot.figure(4)
    n_gram = stat.n_gram(songs, n)
    plot.plot(n_gram.values())
    plot.xlabel("{} - grams".format(n))
    plot.ylabel("frequency")
    plot.draw()
    print(n_gram)


def plot_interval_n_gram(n):
    plot.figure(5)
    interval_n_gram = stat.interval_n_gram(songs, n)
    plot.plot(interval_n_gram.values())
    plot.xlabel("{} - interval grams".format(n))
    plot.ylabel("frequency")
    plot.draw()


plot_histogram()
plot_following_pitches()
plot_intervals()
plot_n_gram(4)  # zipf's law? (1/x)
plot_interval_n_gram(4)
plot.show()
blubb = 0
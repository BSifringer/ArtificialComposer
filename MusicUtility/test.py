import MusicUtility.LoadStore as ls
import MusicUtility.Statistics as stat
import matplotlib.pyplot as plot

songs = ls.load_pkl("C:\\Users\\NiWa\\PycharmProjects\\ArtificialComposer\\Monophonic\\BobSturm.pkl")
# ls.store_midi(songs[1],path="C:/Users/NiWa/Desktop/",label="test") # store a song

plot.figure(1)
pitch_histogram = stat.pitch_histogram(songs, stat.pitches)
plot.plot(pitch_histogram.keys(), pitch_histogram.values())
plot.xlabel("key")
plot.ylabel("frequency")
plot.draw()

plot.figure(2)
following_pitches_histogram = stat.following_pitches_histogram(songs, stat.pitches)
plot.imshow([[following_pitches_histogram[(i, j)] for i in stat.pitches] for j in stat.pitches], origin='lower')
plot.xlabel("key 1")
plot.ylabel("key 2")
plot.draw()

plot.figure(3)
interval_histogram = stat.interval_histogram(songs, stat.pitches)
plot.plot(interval_histogram.keys(), interval_histogram.values())
plot.xlabel("interval")
plot.ylabel("frequency")
plot.show()

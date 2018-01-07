import collections
import numpy as np


def pitch_histogram(songs, pitches, note_length=None):
    """ histogram-dictionary of played pitches in songs. 0th order n-gram """

    histogram = {pitch: 0 for pitch in pitches}

    for song in songs:
        for i in range(len(song.pitch)):
            if note_length is None:
                histogram[song.pitch[i]] = histogram[song.pitch[i]] + 1
            elif song.t[i] == note_length:
                histogram[song.pitch[i]] = histogram[song.pitch[i]] + 1

    return histogram


def following_pitches_histogram(songs, pitches):
    """ histogram-dictionary of 2 following pitches in songs. 1st order n-gram """

    histogram = {(p1, p2): 0 for p1 in pitches for p2 in pitches}

    for song in songs:
        for i in range(len(song.pitch)-1):
            histogram[(song.pitch[i], song.pitch[i+1])] = histogram[(song.pitch[i], song.pitch[i+1])]+1

    return histogram


def n_gram(songs, n):
    """ n-gram dictionary of following pitches """

    histogram = {}
    n = n+1

    for song in songs:
        for i in range(len(song.pitch)-n):
            tmp = []
            for j in range(n):
                # tmp.append(dictionaries["pitch_text"].index(song.pitch[i+j]))
                tmp.append(song.pitch[i + j])

            key = tuple(tmp)
            if key in histogram.keys():
                histogram[key] = histogram[key]+1
            else:
                histogram[key] = 1

    return collections.OrderedDict(sorted(histogram.items(), key=lambda x: x[1], reverse=True))


def interval_n_gram(songs, n):
    """ n-gram dictionary of following intervals """

    histogram = {}

    for song in songs:
        for i in range(len(song.pitch)-n-1):
            tmp = []
            for j in range(n):
                tmp.append(song.pitch[i + j + 1]-song.pitch[i + j])

            key = tuple(tmp)
            if key in histogram.keys():
                histogram[key] = histogram[key]+1
            else:
                histogram[key] = 1

    return collections.OrderedDict(sorted(histogram.items(), key=lambda x: x[1], reverse=True))


def interval_histogram(songs, pitches, note_length=None):
    """ histogram-dictionary of intervals in songs """

    histogram = {interval: 0 for interval in range(min(pitches)-max(pitches), max(pitches)-min(pitches)+1)}
    histogram = collections.OrderedDict(sorted(histogram.items()))

    for song in songs:
        for i in range(len(song.pitch)-1):
            if note_length is None:
                histogram[song.pitch[i+1]-song.pitch[i]] = histogram[song.pitch[i+1]-song.pitch[i]]+1
            elif song.t[i+1] == note_length:
                histogram[song.pitch[i + 1] - song.pitch[i]] = histogram[song.pitch[i + 1] - song.pitch[i]] + 1

    return histogram


def following_intervals_histogram(songs, pitches):
    """ histogram-dictionary of 2 following pitches in songs. 1st order n-gram """

    histogram = {(p1, p2): 0 for p1 in range(min(pitches)-max(pitches), max(pitches)-min(pitches)+1)
                 for p2 in range(min(pitches)-max(pitches), max(pitches)-min(pitches)+1)}

    for song in songs:
        for i in range(len(song.pitch)-2):
            histogram[(song.pitch[i+1]-song.pitch[i], song.pitch[i+2]-song.pitch[i+1])] = \
                histogram[(song.pitch[i+1]-song.pitch[i], song.pitch[i+2]-song.pitch[i+1])]+1

    return histogram


def song_length_histogram(songs, max_song_length):
    """ histogram of song lengths """

    histogram = {length: 0 for length in range(max_song_length)}
    histogram = collections.OrderedDict(sorted(histogram.items()))

    for song in [s for s in songs if len(s.pitch) in range(max_song_length)]:
        histogram[len(song.pitch)] = histogram[len(song.pitch)]+1

    return histogram


def normalize(values):
    """ return normalized values """
    values = np.asarray(list(values))
    return values/np.sum(values)


def entropy(values):
    """ return entropy of values"""
    values = normalize(values)
    return -sum(values*np.log2(values))


def n_gram_entropy(songs, max_n):
    """ return entropy of n-grams up to max_n"""
    n_entropy = {length: 0 for length in range(max_n)}
    n_entropy = collections.OrderedDict(sorted(n_entropy.items()))

    for i in range(max_n):
        i_gram = n_gram(songs, i)
        n_entropy[i] = entropy(i_gram.values())

    return n_entropy


def interval_n_gram_entropy(songs, max_n):
    """ return interval entropy of n-grams up to max_n"""
    n_entropy = {length: 0 for length in range(max_n)}
    n_entropy = collections.OrderedDict(sorted(n_entropy.items()))

    for i in range(max_n):
        i_gram = interval_n_gram(songs, i)
        n_entropy[i] = entropy(i_gram.values())

    return n_entropy

import collections


def pitch_histogram(songs, pitches):
    """ histogram-dictionary of played pitches in songs"""

    histogram = {pitch: 0 for pitch in pitches}

    for song in songs:
        for pitch in song.pitch:
            histogram[pitch] = histogram[pitch]+1

    return histogram


def following_pitches_histogram(songs, pitches):
    """ histogram-dictionary of 2 following pitches in songs """

    histogram = {(p1, p2): 0 for p1 in pitches for p2 in pitches}

    for song in songs:
        for i in range(len(song.pitch)-1):
            histogram[(song.pitch[i], song.pitch[i+1])] = histogram[(song.pitch[i], song.pitch[i+1])]+1

    return histogram


def interval_histogram(songs, pitches):
    """ histogram-dictionary of intervals in songs """

    histogram = {interval: 0 for interval in range(min(pitches)-max(pitches), max(pitches)-min(pitches)+1)}
    histogram = collections.OrderedDict(sorted(histogram.items()))

    for song in songs:
        for i in range(len(song.pitch)-1):
            histogram[song.pitch[i+1]-song.pitch[i]] = histogram[song.pitch[i+1]-song.pitch[i]]+1

    return histogram


# pitches found in "BobSturm.pkl"
pitches = range(48, 96)

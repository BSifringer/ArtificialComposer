import numpy as np


def generate_songs_2_pitch_indices(songs):
    """ generate 2 dictionaries to map pitches to integer values and vice versa
        this is needed to generate the input_vector (this function takes an integer song)"""
    pitches = set.union(*[set(s.pitch) for s in songs])
    pitch2index = {p: i for i, p in enumerate(pitches)}
    index2pitch = {i: p for i, p in enumerate(pitches)}
    return pitch2index, index2pitch


def generate_songs_2_beat_indices(songs):
    """ generate 2 dictionaries to map beats to integer values and vice versa
        this is needed to generate the input_vector (this function takes an integer song)"""
    beats = set.union(*[set(s.t) for s in songs])
    beat2index = {b: i for i, b in enumerate(beats)}
    index2beat = {i: b for i, b in enumerate(beats)}
    return beat2index, index2beat


def generate_input_vector_pitch_with_ending(songs, songs_length, pitches_length):
    """ convert integer song to first layer input of shape len(songs) x songs_length x (pitches + 1)_length
        the size of last input dimension is increased by 1 in order to have an ending
        use this method only directly before feeding data into the training-procedure in order to save storage """
    return_input_vector = np.zeros([len(songs), songs_length, (pitches_length+1)])
    for i in range(len(songs)):
        for j in range(min(len(songs[i].pitch), songs_length)):
            if songs[i].pitch[j] in range(pitches_length):
                return_input_vector[i, j, songs[i].pitch[j]] = 1
        if len(songs[i].pitch) in range(songs_length):
            return_input_vector[i, len(songs[i].pitch), pitches_length] = 1
    return return_input_vector


def generate_input_vector_pitch_and_beat_with_ending(songs, songs_length, pitches_length, beats_length):
    """ convert integer song to first layer input of shape len(songs) x songs_length x (pitches + beats + 1)_length
        the size of last input dimension is increased by 1 in order to have an ending
        use this method only directly before feeding data into the training-procedure in order to save storage """
    return_input_vector = np.zeros([len(songs), songs_length, (pitches_length+beats_length+1)])
    for i in range(len(songs)):
        for j in range(min(len(songs[i].pitch), songs_length)):
            if songs[i].pitch[j] in range(pitches_length):
                return_input_vector[i, j, songs[i].pitch[j]] = 1
            if songs[i].t[j] in range(beats_length):
                return_input_vector[i, j, songs[i].pitch[j]+pitches_length] = 1
        if len(songs[i].pitch) in range(songs_length):
            return_input_vector[i, len(songs[i].pitch), pitches_length+beats_length] = 1
    return return_input_vector


def generate_output_vector_pitch_with_ending(songs, songs_length, pitches_length):
    """ convert integer song to first layer input of shape len(songs) x songs_length x (pitches + 1)_length
            the size of last input dimension is increased by 1 in order to have an ending
            use this method only directly before feeding data into the training-procedure in order to save storage """
    return_output_vector = np.zeros([len(songs), songs_length, (pitches_length + 1)])
    for i in range(len(songs)):
        for j in range(1, min(len(songs[i].pitch), songs_length + 1)):
            if songs[i].pitch[j] in range(pitches_length):
                return_output_vector[i, j-1, songs[i].pitch[j]] = 1
        if len(songs[i].pitch)+1 in range(songs_length):
            return_output_vector[i, len(songs[i].pitch)-1, pitches_length] = 1
    return return_output_vector

import midi
import _pickle as cPickle
import MusicUtility.Song as Song


def store_midi(song, path="", label="", tag="retrieved", resolution=192):
    """ store song in midi file"""

    # Instantiate a MIDI Pattern (contains a list of tracks)
    pattern = midi.Pattern(format=0, resolution=resolution)
    # Instantiate a MIDI Track (contains a list of MIDI events)
    track = midi.Track()
    # Append the track to the pattern
    pattern.append(track)
    tick = 0
    events = []

    # if dtSeq is None assume file is monophonic

    if song.dt is None:
        for T, p in zip(song.t, song.pitch):
            events.append({'t': tick, 'p': p, 'm': 'ON'})
            tick = tick + int(T*resolution)
            events.append({'t': tick, 'p': p, 'm': 'OFF'})
    else:
        for T, p, dt in zip(song.dt, song.pitch, song.dt):
            events.append({'t': tick, 'p': p, 'm': 'ON'})
            tick = tick + int(dt*resolution)
            events.append({'t': tick, 'p': p, 'm': 'OFF'})

    events = sorted(events, key=lambda k: k['t'])
    tick = 0
    for event in events:
        if event['m'] == 'ON':
            e = midi.NoteOnEvent(tick=event['t']-tick, velocity=90, pitch=event['p'])
        if event['m'] == 'OFF':
            e = midi.NoteOffEvent(tick=event['t']-tick, velocity=90, pitch=event['p'])
        track.append(e)
        tick = event['t']

    # Add the end of track event, append it to the track
    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)
    # Save the pattern to disk
    midi.write_midifile(path+label+"_"+tag+".mid", pattern)
    return pattern


def load_pkl(filename):
    """ load songs from pkl file """

    f = open(filename, 'rb')
    data = cPickle.load(f)
    f.close()
    return [Song.Song(t, pitch, dt) for t, pitch, dt in zip(data[0]['Tseqs'], data[0]['pitchseqs'], data[0]['dtseqs'])]

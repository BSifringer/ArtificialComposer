import midi
# install from git: pip install git+https://github.com/vishnubob/python-midi@feature/python3
import _pickle as cPickle
import numpy as np

def writeMIDI(Tseq, pitchseq, path="", label="", tag="retrieved", resolution=192):
	# Instantiate a MIDI Pattern (contains a list of tracks)
	pattern = midi.Pattern(format = 0, resolution = resolution)
	# Instantiate a MIDI Track (contains a list of MIDI events)
	track = midi.Track()
	# Append the track to the pattern
	pattern.append(track)
	tick = 0
	Events = []

	for T, p in zip(Tseq, pitchseq):
		tick = tick + int(T*resolution)
		Events.append({'t': tick, 'p': p, 'm': 'ON'})
		Events.append({'t': tick+int(T*resolution), 'p': p, 'm': 'OFF'})

	Events = sorted(Events, key=lambda k: k['t'])
	tick = 0
	for event in Events:
		if event['m'] == 'ON':	
			e =  midi.NoteOnEvent(tick=event['t']-tick, velocity=90, pitch=event['p'])
		if event['m'] == 'OFF':	
			e =  midi.NoteOffEvent(tick=event['t']-tick, velocity=90, pitch=event['p'])
		track.append(e)
		tick = event['t']

	# Add the end of track event, append it to the track
	eot = midi.EndOfTrackEvent(tick=1)
	track.append(eot)
	# Save the pattern to disk
	midi.write_midifile(path+label+"_"+tag+".mid", pattern)
	return pattern

def load(filename):
	f = open(filename, 'rb')
	loaded_params = cPickle.load(f)
	f.close()
	return loaded_params

def randomtranspose(pitch_seq):
	upperbound = max(pitch_seq)
	lowerbound = min(pitch_seq)
	possible_shifts = range(MIN_DATA_PITCH-lowerbound, MAX_DATA_PITCH-upperbound)
	if len(possible_shifts) == 0:
		possible_shifts.append(0)
	shift = np.random.choice(possible_shifts)
	out = [x+shift for x in pitch_seq]
	return out

if __name__ == '__main__':

	data, dictionaries = load("BobSturm.pkl")

	#Look at the dictionaries
	print(dictionaries)
	MIN_DATA_PITCH = dictionaries["pitchseqs"][0]
	MAX_DATA_PITCH = dictionaries["pitchseqs"][-1]

	#Select one example
	idx = 0

	#Write back a MIDI file
	writeMIDI(data["Tseqs"][idx], data["pitchseqs"][idx], label=str(idx))
	writeMIDI(data["Tseqs"][idx], randomtranspose(data["pitchseqs"][idx]), label=str(idx)+"_transposed")

	#Look at the data 
	for p,T,dt in zip(data["pitchseqs"][idx], data["Tseqs"][idx], data["dtseqs"][idx]):
		print(p,T,dt)

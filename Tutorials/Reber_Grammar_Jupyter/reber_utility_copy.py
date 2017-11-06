import random
from neupy.datasets import make_reber



def generate_continuous_reberwords(number,minimumlength):
	D=[]
	for i in range(number):
		dataset=''
		while len(dataset) < minimumlength: 
			embeddedStep = random.choice('TP') 
			dataset += 'B' + embeddedStep + make_reber(1)[0] + embeddedStep + 'E'
		D.append(dataset)
	return D

char2index = {'B': 0, 'E': 1, 'P': 2, 'S': 3, 'T': 4, 'V': 5, 'X': 6}
index2char = dict((i, c) for i, c in enumerate(char2index))



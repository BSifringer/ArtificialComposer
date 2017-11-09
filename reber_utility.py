import random
import numpy as np


class State_machine:

    current_state = 'A'
    # example: states: A,B,C -> transitions to A,B,C
    # each transition contains probability + label (X,Y,Z)
    # probabilities have to sum up to 1
    state_transitions = ({
        'A': {'A': [0.1, 'X'],'B': [0.2, 'Y'], 'C': [0.7, 'Z']},
        'B': {'A': [0.5, 'X'],'B': [0.0, 'Y'], 'C': [0.5, 'Z']},
        'C': {'A': [0.0, 'X'],'B': [0.9, 'Y'], 'C': [0.1, 'Z']}
    })

    def __init__(self,current_state = None,state_transitions = None):

        if current_state is not None:
            self.current_state = current_state
        if state_transitions is not None:
            self.state_transitions = state_transitions
        self.transitions = []
        for i in self.state_transitions.itervalues():
            for j in i.itervalues():
                self.transitions.append(j[1])
        self.transitions = np.unique(np.asarray(self.transitions))
        # print(np.asarray([i for i in [j.values() for j in self.state_transitions.values()]]))
        #self.transitions = np.unique(np.asarray([i for i in [j.values() for j in self.state_transitions.values()]])[:,:,1])
        self.trans2index = {self.transitions[i]:i for i in range(len(self.transitions))}
        self.index2trans = dict((i, c) for i, c in enumerate(self.trans2index))
        eye = np.eye(len(self.transitions))
        self.trans2vector = {self.transitions[i]:eye[i,:] for i in range(len(self.transitions))}

    def to_transitions(self,vectors):
        return np.asarray([self.index2trans[i] for i in np.argmax(vectors,axis=len(vectors.shape)-1)])

    def to_vectors(self,transitions):
        return np.asarray([self.trans2vector[i] for i in transitions])

    def to_X_and_Y(self,words):
        X = np.asarray([np.ndarray.tolist(self.to_vectors(w)) for w in words])
        Y = np.roll(X, -1, axis=1)
        X = np.delete(X, -1, axis=1)
        Y = np.delete(Y, -1, axis=1)
        return X,Y

    def make_step(self):
        rand = random.uniform(0,1)
        lower_bound = 0
        for (k,v) in self.state_transitions[self.current_state].iteritems():
            if lower_bound <= rand <= lower_bound+v[0]:
                self.current_state = k
                return v[1]
            else:
                lower_bound += v[0]
        raise Exception("state transitions probabilities don't sum up to 1!")

    def make_steps(self,start_state=None,end_state=None,min_steps=None):
        if start_state!=None:
            self.current_state=start_state
        word = [self.make_step()]
        num_steps = 1
        while True:
            if(end_state==None or self.current_state==end_state)and(min_steps==None or num_steps>=min_steps):
                return np.asarray(word)
            word.append(self.make_step())
            num_steps+=1

    def make_words(self,num_words=1,start_state=None,end_state=None,min_steps=None):
        #return np.asarray([np.ndarray.tolist(self.make_steps(start_state, end_state, min_steps)) for i in range(num_words)])
        return [self.make_steps(start_state,end_state,min_steps) for i in range(num_words)]

def generate_reber_machine_discrete():
    current_state = 'A1'
    state_transitions = ({
        'Start': {'A1':[0.5,'T'],'C1':[0.5,'V']},
        'End': {'Start':[1,'O']},
        'A1': {'A1':[0.8,'P'],'B1' :[0.2,'T']},
        'B1': {'C1':[0.8,'X'],'End':[0.2,'S']},
        'C1': {'C1':[0.8,'X'],'D1' :[0.2,'V']},
        'D1': {'B1':[0.8,'P'],'End':[0.2,'S']}
    })
    return State_machine(current_state,state_transitions)

def generate_memory_machine():
    current_state = 'Start'
    state_transitions = ({
        'Start':{'AW':[0.25,'A'],'BW':[0.25,'B'],'CW':[0.25,'C'],'DW':[0.25,'D']},
        'AW':{'AW':[0.9,'X'],'AR':[0.1,'R']},
        'AR':{'End':[1,'A']},
        'BW':{'BW':[0.9,'X'],'BR':[0.1,'R']},
        'BR':{'End':[1,'B']},
        'CW':{'CW':[0.9,'X'],'CR':[0.1,'R']},
        'CR':{'End':[1,'C']},
        'DW':{'DW':[0.9,'X'],'DR':[0.1,'R']},
        'DR':{'End':[1,'D']},
        'End':{'Start':[1,'W']}
    })
    return State_machine(current_state,state_transitions)
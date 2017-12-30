class Song:

    def __init__(self, t, pitch, dt=None):
        self.t = t
        self.pitch = pitch
        self.dt = dt
        self.size = len(pitch)

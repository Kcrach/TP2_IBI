import numpy
import random
from collections import namedtuple

Interaction = namedtuple('Interaction',
                        ('state', 'action', 'next_state', 'reward', 'endOfEp'))

class Memory(object):
    def __init__(self):
        self.capacity = 10000
        self.memoryReplay = []
    
    def pushMemory(self, state, action, nextState, reward, endOfEp):
        if len(self.memoryReplay) > self.capacity :
            # Remove first element
            self.memoryReplay.pop(0)
        # Add last interaction into buffer
        self.memoryReplay.append(Interaction(state, action, nextState, reward, endOfEp))
    
    def sampling(self, size):
        # Return a batch of random data into the buffer, batch of size "size"
        return random.sample(self.memoryReplay, size)

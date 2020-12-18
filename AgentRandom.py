from Network import Network
import random
import numpy
import torch

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space, ob_space):
        self.action_space = action_space

        self.network = Network(ob_space.shape[0], [16, 32 , 16], action_space.n)
        self.t_network = Network(ob_space.shape[0], [16, 32 , 16], action_space.n)

    def act(self, observation, epsilon):
        # Convert to tensor
        ob = torch.from_numpy(observation).float().unsqueeze(0)
        # Switch to "evaluate mode"
        self.network.eval()

        # With no gradient
        with torch.no_grad():
            # Calc Q Val for observation
            qval = self.network(ob)

        # Switch to "training mode"
        self.network.train()   

        #Greedy
        # Draw random in [0,1]
        rand = random.random() 
        if rand < (1 - epsilon):
            # Do best action
            return numpy.argmax(qval.data.numpy())
        else:
            # Do random action : choose action in [0, |action_space|]
            return random.choice(numpy.arange(self.action_space.n))
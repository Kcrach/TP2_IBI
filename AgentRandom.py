    
import argparse
import sys

import gym
import numpy
import random
from matplotlib import pyplot
from gym import wrappers, logger
from collections import namedtuple
import torch

Interaction = namedtuple('Interaction',
                        ('state', 'action', 'next_state', 'reward', 'enfOfEp'))

class Memory(object):
    def __init__(self):
        self.capacity = 100000
        self.memoryReplay = []
        self.pos = 0
    
    def pushMemory(self, state, action, nextState, reward, endOfEp):
        if len(self.memoryReplay) == self.capacity :
            # Remove first element
            self.memoryReplay = numpy.delete(self.memoryReplay, 0)
        # Add last interaction into buffer
        self.memoryReplay.append(Interaction(state, action, nextState, reward, endOfEp))
        self.pos = (self.pos + 1) % self.capacity
    
    def sampling(self, size):
        # Return a batch of random data into the buffer, batch of size "size"
        return random.sample(self.memoryReplay, size)
        
class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

class Network(torch.nn.Module):
    def __init__(self, d_in, h_tab, d_out, f=torch.nn.Sigmoid):
        super(Network, self).__init__()
        self.fun = f
        if len(h_tab) == 0:
            linear = [torch.nn.Linear(d_in, d_out, bias=False)]
        else:
            linear = list()
            linear.append(torch.nn.Linear(d_in, h_tab[0], bias=True))
            linear.append(self.fun())
            for h in range(len(h_tab) - 1):
                next_h = h + 1
                linear.append(torch.nn.Linear(h_tab[h], h_tab[next_h],
                                              bias=True))
                linear.append(self.fun())
            linear.append(torch.nn.Linear(h_tab[-1], d_out, bias=True))
            linear.append(torch.nn.Softmax(dim=1))
        self.param = torch.nn.ModuleList(linear)

    def forward(self, x_v):
        for f in self.param:
            x_v = f(x_v)
        return x_v


if __name__ == '__main__':
    eta = 0.0001

    parser = argparse.ArgumentParser(description=None)
    # Question 1
    parser.add_argument('env_id', nargs='?', default='CartPole-v1', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make(args.env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = RandomAgent(env.action_space)

    mem = Memory()

    episode_count = 100
    reward = 0
    done = False

    rewardPerEp = numpy.array([])
    ep = numpy.array([])
    interaction = numpy.array([])

    # Instantiate the NN with 3 hidden layers
    ob = env.reset()
    model = Network(ob.shape[0], [16, 32, 16], env.action_space.n, torch.nn.Tanh)
    model = model.double()

    for i in range(episode_count):
        ob = env.reset()
        nbInteraction = 0
        sumReward = 0
        while True:
            nbInteraction += 1
            # State before that the agent do the action
            actualState = ob

            print(torch.from_numpy(actualState).double())
            qVal = model(torch.from_numpy(actualState).double())     

            action = agent.act(ob, reward, done)

            ob, reward, done, _ = env.step(action)
            sumReward += reward

            if done:
                # Data to draw graphics
                rewardPerEp = numpy.append(rewardPerEp, sumReward)
                ep = numpy.append(ep, i)
                interaction = numpy.append(interaction, nbInteraction)

                # Write logs
                print("Episode " + str(i) + " : " + str(sumReward) + " in " + str(nbInteraction) + ".")

                # Add the interaction in memory
                mem.pushMemory(actualState, action, ob, reward, True)
                break
            else:
                # Add the interaction in memory
                mem.pushMemory(actualState, action, ob, reward, False)

            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.
    
    # Graphic reward per episode / episode
    pyplot.plot(ep, rewardPerEp)
    pyplot.gca().set_ylabel("Récompense totale")
    pyplot.gca().set_xlabel("Épisode")
    pyplot.title("Récompense totale par épisode")
    pyplot.show()

    # Graphic total reward / nb interaction
    pyplot.plot(interaction, rewardPerEp)
    pyplot.gca().set_ylabel("Récompense totale")
    pyplot.gca().set_xlabel("Nb Interaction")
    pyplot.title("Récompense totale par rapport au nombre d'interaction")
    pyplot.show()

    # Close the env and write monitor result info to disk
    env.close()
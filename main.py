    
import argparse
import sys

import gym
import numpy
from matplotlib import pyplot
from gym import wrappers, logger
import torch
from MemoryReplay import Memory
from AgentRandom import RandomAgent

if __name__ == '__main__':
    # Hyper-parameters
    eta = 0.0001
    batch_size = 64
    gamma = 0.999
    eps_start = 0.9
    eps_end = 0.05
    eps_decay = 200

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
    agent = RandomAgent(env.action_space, env.observation_space, eta)

    mem = Memory()

    episode_count = 100
    reward = 0
    done = False

    rewardPerEp = numpy.array([])
    ep = numpy.array([])
    interaction = numpy.array([])

    for i in range(episode_count):
        ob = env.reset()
        nbInteraction = 0
        sumReward = 0

        while True:
            nbInteraction += 1
            # State before that the agent do the action
            actualState = ob
            action = agent.act(ob, eps_end, eps_start, eps_decay)

            ob, reward, done, _ = env.step(action)
            sumReward += reward
                
            # Add the interaction in memory
            mem.pushMemory(actualState, action, ob, reward, done)

            # Verify that the size of the sample is inferior than the size of the memory
            if batch_size <= len(mem.memoryReplay):
                # After we can learn
                sample_exp = mem.sampling(batch_size)
                agent.learn(sample_exp,gamma)

            if done:
                # Data to draw graphics
                rewardPerEp = numpy.append(rewardPerEp, sumReward)
                ep = numpy.append(ep, i)

                interaction = numpy.append(interaction, nbInteraction)
                break

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
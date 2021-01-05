    
import argparse
import sys

import gym
import numpy
import vizdoomgym
import skimage.transform
import skimage.color
from matplotlib import pyplot
from gym import wrappers, logger
import torch
from MemoryReplay import Memory
from AgentRandom import RandomAgent
from AgentVizdoom import VizdoomAgent
#from gym.wrappers import FrameStack

def preprocess(img, reso):
    img = skimage.transform.resize(img, reso)
    img = skimage.color.rgb2gray(img)
    img = img.astype(numpy.float32)
    img = img.reshape([1, 1, int(reso[0]), int(reso[1])])
    return img

if __name__ == '__main__':
    # Hyper-parameters
    environment_used = "VizdoomBasic-v0" #|| "CartPole-v1"
    eta = 0.0005
    batch_size = 64
    gamma = 0.999
    epsilon_start = 1
    epsilon_end = 0.05
    epsilon_decay = 0.998
    update_freq = 200

    # CNN
    resolution = [112, 64]
    k_size = 5
    stride = 2

    parser = argparse.ArgumentParser(description=None)
    # Question 1
    # Change the var called environment_used to change the environment
    parser.add_argument('env_id', nargs='?', default=environment_used, help='Select the environment to run')
    if environment_used == "VizdoomBasic-v0":
        parser.add_argument('depth', nargs='?', default=True)
        parser.add_argument('labels', nargs='?', default=True)
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

    if environment_used == "VizdoomBasic-v0":
        agent = VizdoomAgent(env.action_space, resolution , eta)
    else:
        agent = RandomAgent(env.action_space, env.observation_space, eta)

    mem = Memory()

    episode_count = 300
    reward = 0
    done = False

    rewardPerEp = numpy.array([])
    ep = numpy.array([])
    interaction = numpy.array([])

    for i in range(episode_count):
        env.reset()
        # Preprocess for CNN
        if environment_used == "VizdoomBasic-v0":
            actual_state, reward, done, info = env.step(env.action_space.sample())
            actual_state=preprocess(actual_state[0], resolution) 

        nbInteraction = 0
        sumReward = 0

        while True:
            nbInteraction += 1
            # Choose the action to do
            action = agent.act(actual_state, epsilon_start)

            # Calculate next state, reward and if the episode is finished or not
            next_state, reward, done, _ = env.step(action)

            if environment_used == "VizdoomBasic-v0":
                next_state = preprocess(next_state[0], resolution)

            sumReward += reward
                
            # Add the interaction in memory
            mem.pushMemory(actual_state, action, next_state, reward, done)
            actual_state = next_state

            # Verify that the size of the sample is inferior than the size of the memory
            if batch_size <= len(mem.memoryReplay):
                # After we can learn
                sample_exp = mem.sample(batch_size)
                agent.opti_model(sample_exp, gamma, update_freq)

            if done:
                # Data to draw graphics
                rewardPerEp = numpy.append(rewardPerEp, sumReward)
                ep = numpy.append(ep, i)

                interaction = numpy.append(interaction, nbInteraction)
                break

            # Decrease the epsilon, at the beginning, the epsilon is high, so the agent will diversify its actions
            # But at the end, the epsilon is low, so the agent will intensify.
            if epsilon_start > epsilon_end :
                epsilon_start *= epsilon_decay

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
    
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
from gym.wrappers import FrameStack, ResizeObservation, GrayScaleObservation

if __name__ == '__main__':
    # Hyper-parameters
    environment_used =  "VizdoomCorridor-v0" # || "CartPole-v1" "VizdoomBasic-v0"
    eta = 0.001 # || 0.01 || 0.0001
    batch_size = 32 # || 256 || 128 || 64 || 2
    gamma = 0.999 # || 0.2 || 0.9
    epsilon_start = 1
    epsilon_end = 0.05
    epsilon_decay = 0.998
    update_freq = 500

    # CNN
    resolution = [112, 64]

    # Test mode : Switch to True to test the learning (without training)
    test_mode = False
    if test_mode:
        # When we test, we intensify
        epsilon_start = 0.0

    parser = argparse.ArgumentParser(description=None)
    # Question 1
    # Change the var called environment_used to change the environment
    parser.add_argument('env_id', nargs='?', default=environment_used, help='Select the environment to run')
    args = parser.parse_args()
    
    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make(args.env_id)

    if environment_used != "CartPole-v1":
        env = GrayScaleObservation(env)
        env = ResizeObservation(env, resolution)
        env = FrameStack(env, 4)
        agent = VizdoomAgent(env.action_space, resolution , eta, test_mode, environment_used)
    else:
        agent = RandomAgent(env.action_space, env.observation_space, eta, test_mode, environment_used)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)

    mem = Memory()

    episode_count = 50
    reward = 0
    done = False

    rewardPerEp = numpy.array([])
    ep = numpy.array([])
    interaction = numpy.array([])

    for i in range(episode_count):
        env.reset()
        actual_state, reward, done, info = env.step(env.action_space.sample())

        nbInteraction = 0
        sumReward = 0

        while True:
            nbInteraction += 1
            # Choose the action to do
            action = agent.act(actual_state, epsilon_start)

            # Calculate next state, reward and if the episode is finished or not
            next_state, reward, done, _ = env.step(action)

            sumReward += reward
            
            if test_mode == False:
                # Add the interaction in memory
                mem.pushMemory(actual_state, action, next_state, reward, done)
            
                if batch_size <= len(mem.memoryReplay):
                    # After we can learn
                    sample_exp = mem.sample(batch_size)
                    agent.opti_model(sample_exp, gamma, update_freq)

                # Decrease the epsilon, at the beginning, the epsilon is high, so the agent will diversify its actions
                # But at the end, the epsilon is low, so the agent will intensify.
                if epsilon_start > epsilon_end :
                    epsilon_start *= epsilon_decay

            actual_state = next_state

            # Display render if we are in test_mode
            if test_mode == True:
                env.render()

            if done:
                # Data to draw graphics
                rewardPerEp = numpy.append(rewardPerEp, sumReward)
                ep = numpy.append(ep, i)

                interaction = numpy.append(interaction, nbInteraction)
                break

            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.
    
    # Save weights of the network used, decomment to save a new Neural Net
    #if test_mode == False:
    #    torch.save(agent.net.state_dict(), "net/" + environment_used + ".pt")

    # Graphic reward per episode / episode
    pyplot.plot(ep, rewardPerEp)
    pyplot.gca().set_ylabel("Récompense totale")
    pyplot.gca().set_xlabel("Épisode")
    pyplot.title("Récompense totale par épisode")
    pyplot.show()


    # Close the env and write monitor result info to disk
    env.close()
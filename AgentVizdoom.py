from Network import CNN
import random
import numpy
import torch
import math

class VizdoomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space, reso, eta, test_mode, environment_used):
        self.action_space = action_space
        self.eta = eta
        self.steps_done = 0

        self.net = CNN(reso, action_space.n)
        self.target_net = CNN(reso, action_space.n)

        # Load weights of the network from file (to test)
        if test_mode:
            self.net.load_state_dict(torch.load("net/" + environment_used + ".pt"))

        # Switch to "evaluate mode"
        self.net.eval()
        self.target_net.eval()

        # self.optimizer = torch.optim.SGD(self.net.parameters(), lr=eta)
        # Adam seems to be better
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=eta)

    def act(self, observation, eps):
        """
            Choose an action with the strategy greedy. 
            With a probability of 1 - eps, the agent intensify (it means that it will do the "best" action).
            With a probability of eps, the agent diversify (random).
        """
        self.steps_done += 1
        # We have [4, 112, 64, 1] But we want [1, 4, 112, 64]
        observation = numpy.swapaxes(observation, 0, 1)
        observation = numpy.swapaxes(observation, 2, 3)
        observation = numpy.swapaxes(observation, 0, 2)

        # Convert to tensor
        ob = torch.from_numpy(observation).float()

        # With no gradient
        with torch.no_grad():
            # Calc Q Val for observation
            qval = self.net(ob)

        #Greedy
        # Draw random in [0,1] 
        rand = random.random() 
        if rand > eps:
            # Do best action
            return int(torch.argmax(qval))
        else:
            # Do random action : choose action in [0, 1, ..., |action_space|] 
            return random.choice(numpy.arange(self.action_space.n))

    def opti_model(self, sample_exp, gamma, update_freq):
        """
            Fonction of learning. The learning is done thanks to the Bellman's equation :
            Q(s,a) = r + (gamma * maxa'(Q'(s',a'))) if the episode is not done.
            Q(s,a) = r if the episode is done.
        """
        self.net.train()
        # Get (state, action, next_state, reward, endOfEp) of each interaction in the sample of experience
        tmp_states = numpy.vstack([[exp.state] for exp in sample_exp if exp is not None])
        states = torch.from_numpy(tmp_states).float()
        tmp_actions = numpy.vstack([exp.action for exp in sample_exp if exp is not None])
        actions = torch.from_numpy(tmp_actions).long()
        tmp_next_states = numpy.vstack([[exp.next_state] for exp in sample_exp if exp is not None])
        next_states = torch.from_numpy(tmp_next_states).float()
        tmp_rewards = numpy.vstack([exp.reward for exp in sample_exp if exp is not None])
        rewards = torch.from_numpy(tmp_rewards).float()
        tmp_endsOfEp = numpy.vstack([exp.endOfEp for exp in sample_exp if exp is not None]).astype(numpy.uint8)
        endsOfEp = torch.from_numpy(tmp_endsOfEp).float()

        loss_func = torch.nn.MSELoss(reduction="sum")

        states = states[:,:,:,:,0]

        prediction = self.net(states).gather(1, actions)

        next_states = next_states[:,:,:,:,0]

        next_qval = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)

        qval = rewards + (gamma * next_qval * (1 - endsOfEp))

        loss = loss_func(prediction, qval)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps_done % update_freq == 0:
            self.update_target()
    
    def update_target(self):
        self.target_net.load_state_dict(self.net.state_dict())

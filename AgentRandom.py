from Network import Network
import random
import numpy
import torch
import math

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space, ob_space, eta):
        self.action_space = action_space
        self.eta = eta
        self.steps_done = 0

        self.net = Network(ob_space.shape[0], [16, 32 , 16], action_space.n)
        self.target_net = Network(ob_space.shape[0], [16, 32 , 16], action_space.n)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=eta)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def act(self, observation, EPS_END, EPS_START, EPS_DECAY):
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        # Convert to tensor
        ob = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
        # Switch to "evaluate mode"
        self.net.eval()

        # With no gradient
        with torch.no_grad():
            # Calc Q Val for observation
            qval = self.net(ob)

        # Switch to "training mode"
        self.net.train()   

        #Greedy
        # Draw random in [0,1] 
        rand = random.random() 
        if rand > eps_threshold:
            # Do best action
            return int(torch.argmax(qval))
        else:
            # Do random action : choose action in [0, |action_space|]
            return random.choice(numpy.arange(self.action_space.n))

    def learn(self, sample_exp, gamma):
        # Get (state, action, next_state, reward, endOfEp) of each interaction in the sample of experience
        states = torch.from_numpy(numpy.vstack([exp.state for exp in sample_exp if exp is not None])).float().to(self.device)
        actions = torch.from_numpy(numpy.vstack([exp.action for exp in sample_exp if exp is not None])).long().to(self.device)
        next_states = torch.from_numpy(numpy.vstack([exp.next_state for exp in sample_exp if exp is not None])).float().to(self.device)
        rewards = torch.from_numpy(numpy.vstack([exp.reward for exp in sample_exp if exp is not None])).float().to(self.device)
        # .astype enable to convert bool -> int, cause we can't convert directly bool -> float
        endsOfEp = torch.from_numpy(numpy.vstack([exp.endOfEp for exp in sample_exp if exp is not None]).astype(numpy.uint8)).float().to(self.device)

        # Establish the loss function
        loss_func = torch.nn.MSELoss()

        self.target_net.eval()

        prediction = self.net(states).gather(1, actions)

        with torch.no_grad():
            next_labels = self.target_net(next_states).max(1)[0].unsqueeze(1)

        # Bellman's equation
        labels = rewards + (gamma * next_labels * (1 - endsOfEp))

        loss = loss_func(prediction, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps_done % 5000 == 0:
            self.update_target()
    
    def update_target(self):
        self.target_net.load_state_dict(self.net.state_dict())

import random
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
from utils import device


class Agent:

    def __init__(self, local_network, target_network, world, memory, config, train_mode=True):
        self.local_network = local_network.to(device)
        self.target_network = target_network.to(device)
        self.world = world
        self.train_mode = train_mode
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.memory = memory
        self.nr_steps = 0
        self.total_reward = 0
        self.finished = False
        self.update_freq = config['update_freq']
        self.eps = 1
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.max_nr_steps = config['max_nr_steps']
        self.algo = config['algo']
        self.output_freq = config['output_freq']
        self.nr_episodes = config['nr_episodes']
        self.eps_min = config['eps_min']
        self.eps_decl = config['eps_decl']
        self.optimizer = optim.Adam(self.local_network.parameters(), lr=config['lr'])
        self.state = None

    def initialize_world(self, eps):
        if self.world.type == 'gym':
            self.state = self.world.env.reset()
        elif self.world.type == 'unity_vector':
            env_info = self.world.env.reset(train_mode=self.train_mode)[self.world.brain_name]
            self.state = env_info.vector_observations[0]
        elif self.world.type == 'unity_visual':
            env_info = self.world.env.reset(train_mode=self.train_mode)[self.world.brain_name]
            self.state = env_info.visual_observations[0]
        self.nr_steps = 0
        self.total_reward = 0
        self.finished = False
        self.eps = eps

    def select_action(self):
        # run inference on the local network
        self.local_network.eval()
        with torch.no_grad():
            if self.world.type == 'unity_visual':
                exp_future_rewards = self.local_network(torch.from_numpy(self.state).permute(0, 3, 1, 2).
                                                        float().to(device))
            else:
                exp_future_rewards = self.local_network(torch.from_numpy(self.state).unsqueeze(0).float().to(device))
        self.local_network.train()

        # apply epsilon greedy action selection
        # what happens if we weigh the expected rewards in stead of the max?
        if random.random() > self.eps:
            return np.argmax(exp_future_rewards.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.world.action_space_size))

    def take_action(self, action):
        if self.world.type == 'gym':
            next_state, reward, done, _ = self.world.env.step(action)
        elif self.world.type == 'unity_vector':
            env_info = self.world.env.step(action)[self.world.brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
        elif self.world.type == 'unity_visual':
            env_info = self.world.env.step(action)[self.world.brain_name]
            next_state = env_info.visual_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
        self.nr_steps += 1
        return next_state, reward, done

    def process_results(self, action, results):
        next_state, reward, done = results
        experience = self.experience(self.state, action, reward, next_state, done)
        self.memory.add(experience)

        if self.nr_steps % self.update_freq == 0:
            if self.memory.enough_data:
                self.update()

        self.total_reward += reward
        self.state = next_state
        if done or self.nr_steps > self.max_nr_steps:
            self.finished = True

    def update(self):
        # get a batch
        states, actions, rewards, next_states, dones = self.memory.sample()
        if self.world.type == 'unity_visual':
            states = states.permute(0, 3, 1, 2)
            next_states = next_states.permute(0, 3, 1, 2)

        # get the q value of the the target using DQN or double DQN

        # ====DQN======================
        if self.algo == 'dqn':
            target_max_q_next_states = self.target_network(next_states).detach().max(1)[0]
            target_q = rewards + self.gamma * target_max_q_next_states.unsqueeze(1) * (1 - dones)

        # =====DOUBLE DQN==============
        elif self.algo == 'double_dqn':
            target_selected_actions = self.local_network(next_states).detach().max(1)[1].unsqueeze(1)
            target_values = self.target_network(next_states).detach().gather(1, target_selected_actions)
            target_q = rewards + self.gamma * target_values.unsqueeze(1) * (1 - dones)

        # get the estimated q value belonging the actions taken
        estimated_local_q = self.local_network(states).gather(1, actions)

        # calculate the loss
        loss = F.mse_loss(target_q, estimated_local_q)

        # update the local network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update the target network
        self.update_target()

    def update_target(self):
        for target_param, local_param in zip(self.target_network.parameters(), self.local_network.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def train(self):
        all_scores = []
        scores = deque(maxlen=self.output_freq)
        eps = 1
        for i in range(1, self.nr_episodes + 1):
            eps = max(self.eps_min, eps*self.eps_decl)
            self.initialize_world(eps)
            while not self.finished:
                a = self.select_action()
                res = self.take_action(a)
                self.process_results(a, res)
            scores.append(self.total_reward)
            all_scores.append(self.total_reward)
            if i % self.output_freq == 0:
                print(f'finished episodes {i-self.output_freq+1} - {i} with avg reward {sum(scores) / len(scores)}')

        return all_scores

    def run(self):
        total_reward = 0
        self.initialize_world(1)
        while not self.finished:
            a = self.select_action()
            next_state, reward, done = self.take_action(a)
            total_reward += reward
            self.state = next_state
            if done:
                self.finished = True
                print(f'Monkey got {total_reward} points!')

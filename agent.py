from collections import namedtuple
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import BananaModel, BananaPixelsModel


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BaseAgent:
    def __init__(self, state_size, action_size, buffer_size=10000, batch_size=256, gamma=0.99, tau=0.001,
                 learning_rate=0.001, update_steps=10, a=0.5):
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_steps = update_steps
        self.global_step = 0
        self.action_size = action_size

        self.qnetwork_local = self.QNetwork(state_size, action_size).to(device)
        self.qnetwork_target = self.QNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)

        #for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
        #    target_param.data.copy_(local_param.data)

        self.replay_buffer = ReplayBuffer(buffer_size, batch_size, a)

    @property
    def QNetwork(self):
        if '_q_class' not in dir(self):
            raise NotImplementedError
        return self._q_class

    def step(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        self.global_step += 1
        if self.global_step % self.update_steps == 0:
            if len(self.replay_buffer) > self.batch_size:
                experiences = self.replay_buffer.sample()
                self.update_local(experiences)

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if np.random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return np.random.choice(np.arange(self.action_size))

    def update_local(self, experiences):
        ids, states, actions, rewards, next_states, dones, weights = experiences

        q_local_0 = self.qnetwork_local.forward(states)
        q_local = q_local_0.gather(-1, actions)

        q_target_1 = self.qnetwork_target.forward(next_states).detach()
        q_local_1 = self.qnetwork_local.forward(next_states).detach()
        next_actions = q_local_1.max(-1, keepdim=True)[1]
        #next_actions = q_target_1.max(-1, keepdim=True)[1]
        q_target = q_target_1.gather(1, next_actions)
        q_target = rewards + ((1. - dones) * self.gamma * q_target)

        loss = F.mse_loss(q_local, q_target, reduction='none')
        weighted_loss = loss * torch.from_numpy(weights).float().to(device)

        self.optimizer.zero_grad()
        computed_loss = loss.cpu().data.numpy()
        weighted_loss.mean().backward()
        self.optimizer.step()

        self.update_target()

        self.replay_buffer.update(ids, computed_loss)

    def update_target(self):
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)


class BananaAgent(BaseAgent):
    _q_class = BananaModel


class BananaPixelAgent(BaseAgent):
    _q_class = BananaPixelsModel


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, a):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.a = a
        self.next_id = 0
        self.memory = []
        self.experience = namedtuple("Experience",
                                     field_names=["id", "state", "action", "reward", "next_state", "done", "loss"])

    def add(self, state, action, reward, next_state, done):
        if len(self.memory) == self.buffer_size:
            self.memory = self.memory[1:]
        if len(self.memory) > 0:
            max_loss = np.max([e.loss for e in self.memory])
        else:
            max_loss = 1.
        self.memory.append(self.experience(self.next_id, state, action, reward, next_state, done, max_loss))
        self.next_id += 1

    def update(self, ids, losses):
        loss_dict = dict(zip(ids[:,0], losses[:,0]))
        for i, e in enumerate(self.memory):
            if e.id in loss_dict:
                self.memory[i] = self.experience(
                    e.id, e.state, e.action, e.reward, e.next_state, e.done, loss_dict[e.id])

    def sample(self):
        losses = np.array([e.loss for e in self.memory], dtype=float)
        p = np.power(losses, self.a)
        p /= np.sum(p)

        experiences = np.random.choice(range(len(self.memory)), size=self.batch_size, replace=False, p=p)
        experiences = [self.memory[i] for i in experiences]

        ids = np.vstack([e.id for e in experiences])
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)
        weights = np.vstack([e.loss for e in experiences])
        weights = 1. / np.power(weights, self.a)
        weights /= np.mean(weights)

        return ids, states, actions, rewards, next_states, dones, weights

    def __len__(self):
        return len(self.memory)

from collections import namedtuple
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import BananaModel, BananaPixelsModel


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BaseAgent:
    def __init__(self, state_size, action_size, buffer_size=10000, batch_size=64, gamma=0.99, tau=0.001,
                 learning_rate=0.001, update_steps=4):
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

        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)

    @property
    def QNetwork(self):
        if '_q_class' not in dir(self):
            raise NotImplementedError
        return self._q_class

    def step(self, state, action, reward, next_state, done, a, b):
        state_t = torch.from_numpy(np.array([state])).float().to(device)
        action_t = torch.from_numpy(np.array([[action]])).long().to(device)
        reward_t = torch.from_numpy(np.array([reward])).float().to(device)
        next_state_t = torch.from_numpy(np.array([next_state])).float().to(device)
        done_t = torch.from_numpy(np.array([done], dtype=np.uint8)).float().to(device)

        self.qnetwork_local.eval()
        with torch.no_grad():
            q_local_0 = self.qnetwork_local.forward(state_t)
            q_local = q_local_0.gather(-1, action_t)
            q_target_1 = self.qnetwork_target.forward(next_state_t)
            next_actions = q_target_1.max(-1, keepdim=True)[1]
            q_target = q_target_1.gather(1, next_actions)
            q_target = reward_t + ((1. - done_t) * self.gamma * q_target)
            error = np.sqrt(F.mse_loss(q_local, q_target, reduction='none').cpu().data.numpy())[0,0]
        self.qnetwork_local.train()

        self.replay_buffer.add(state, action, reward, next_state, done, error)
        self.global_step += 1
        if self.global_step % self.update_steps == 0:
            if len(self.replay_buffer) > self.batch_size:
                experiences = self.replay_buffer.sample(a, b)
                self.update_local(experiences)
                self.update_target()

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

        with torch.no_grad():
            q_target_1 = self.qnetwork_target.forward(next_states).detach()
            q_local_1 = self.qnetwork_local.forward(next_states).detach()
            next_actions = q_local_1.max(-1, keepdim=True)[1]
            q_target = q_target_1.gather(1, next_actions)
            q_target = rewards + ((1. - dones) * self.gamma * q_target)

        q_local_0 = self.qnetwork_local.forward(states)
        q_local = q_local_0.gather(-1, actions)

        loss = F.mse_loss(q_local, q_target, reduction='none')
        with torch.no_grad():
            error = np.sqrt(loss.detach().cpu().data.numpy())
        weighted_loss = loss * weights

        self.optimizer.zero_grad()
        weighted_loss.mean().backward()
        self.optimizer.step()

        self.replay_buffer.update(ids, error)

    def update_target(self):
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)


class BananaAgent(BaseAgent):
    _q_class = BananaModel


class BananaPixelAgent(BaseAgent):
    _q_class = BananaPixelsModel


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.next_id = 0
        self.memory = []
        self.experience = namedtuple("Experience",
                                     field_names=["id", "state", "action", "reward", "next_state", "done", "error"])

    def add(self, state, action, reward, next_state, done, error):
        if len(self.memory) == self.buffer_size:
            self.memory = self.memory[1:]
        self.memory.append(self.experience(self.next_id, state, action, reward, next_state, done, error))
        self.next_id += 1

    def update(self, ids, errors):
        err_dict = dict(zip(ids[:,0], errors[:,0]))
        for i, e in enumerate(self.memory):
            if e.id in err_dict:
                self.memory[i] = self.experience(
                    e.id, e.state, e.action, e.reward, e.next_state, e.done, err_dict[e.id])

    def sample(self, a, b):
        errors = np.array([e.error for e in self.memory], dtype=float)
        p = np.power(errors, a)
        p /= np.sum(p)

        experiences_i = np.random.choice(range(len(self.memory)), size=self.batch_size, replace=False, p=p)
        experiences = [self.memory[i] for i in experiences_i]
        p = np.array([p[i] for i in experiences_i])

        ids = np.vstack([e.id for e in experiences])
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)
        weights = np.power(1 / (len(self.memory) * p), b)
        weights = torch.from_numpy(weights).float().to(device)

        return ids, states, actions, rewards, next_states, dones, weights

    def __len__(self):
        return len(self.memory)

import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from RL.model import LinearQNet

class DQNAgent:
    def __init__(
        self,
        state_size=11,
        action_size=3,
        hidden_size=128,
        lr=1e-3,
        gamma=0.9,
        memory_size=100_000,
        batch_size=1024,
        target_update_every=1000,
        device=None
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_every = target_update_every

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = LinearQNet(state_size, hidden_size, action_size).to(self.device)
        self.target_net = LinearQNet(state_size, hidden_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.memory = deque(maxlen=memory_size)
        self.train_steps = 0

        # epsilon-greedy
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995  # decay each episode

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # ε-greedy
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # current Q(s,a)
        q_pred = self.policy_net(states_t).gather(1, actions_t)

        # target: r + γ * max_a' Q_target(s',a')  (if not done)
        with torch.no_grad():
            q_next = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]
            q_target = rewards_t + (1 - dones_t) * self.gamma * q_next

        loss = self.loss_fn(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def end_episode(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path="Models/snake_dqn.pth"):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path="Models/snake_dqn.pth"):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())

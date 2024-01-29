import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from collections import deque
import itertools
import random

import wandb

GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = 50_000
MIN_REPLAY_SIZE = 1_000 
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY_STEP = 10_000 
TARGET_UPDATE_FREQ = 1_000 
LEARNING_RATE = 5e-4
TOTAL_STEP = 100_000

class DeepQNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, fc1_dims=256, fc2_dims=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
        )

    def forward(self, x):
        return self.model(x)
    
    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        q_values = self(obs_t.unsqueeze(0))

        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()

        return action

class ReplayBuffer:
    def __init__(self, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE):
        self.replay_buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def store_transition(self, transition):
        self.replay_buffer.append(transition)

    def generate_batch(self):
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        return batch

class DQN:
    def __init__(self, input_dims, n_actions, lr=LEARNING_RATE, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE,
                 epsilon_start=EPSILON_START, epsilon_end=EPSILON_END, epsilon_decay_step=EPSILON_DECAY_STEP):
        self.Q_eval = DeepQNetwork(input_dims=input_dims, n_actions=n_actions)
        self.Q_target = DeepQNetwork(input_dims=input_dims, n_actions=n_actions)
        self.optimizer = optim.Adam(self.Q_eval.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size, batch_size=batch_size)

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_step

    def update_target_network(self):
        self.Q_target.load_state_dict(self.Q_eval.state_dict())

    def store_transition(self, transition):
        self.replay_buffer.store_transition(transition=transition)

    def act(self, state):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        rnd_sample = random.random()
        if rnd_sample <= self.epsilon:
            action = env.action_space.sample()
        else:
            action = self.Q_eval.act(state)

        return action

    def learn(self):
        batch = self.replay_buffer.generate_batch()

        obses = np.asanyarray([t[0] for t in batch])
        actions = np.asanyarray([t[1] for t in batch])
        rews = np.asanyarray([t[2] for t in batch])
        dones = np.asanyarray([t[3] for t in batch])
        new_obses = np.asanyarray([t[4] for t in batch])

        obses_t = torch.as_tensor(obses, dtype=torch.float32)
        actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
        rews_t = torch.as_tensor(rews, dtype=torch.float32).unsqueeze(-1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
        new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32)

        # Compute Targets
        target_q_values = self.Q_target(new_obses_t)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

        targets = rews_t + GAMMA * (1 - dones_t) * max_target_q_values

        # Compute Loss
        q_values = self.Q_eval(obses_t)
        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)
        # loss = nn.functional.mse_loss(action_q_values, targets)
        loss = nn.functional.smooth_l1_loss(action_q_values, targets)

        # Gradient Descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
if __name__ == '__main__':
    wandb.init(project="classic-drl-algorithms", entity="cc299792458", name='dqn',
               config={
                "gamma": GAMMA,
                "batch size": BATCH_SIZE,
                "buffer size": BUFFER_SIZE,
                "min replay size": MIN_REPLAY_SIZE, 
                "epsilon start": EPSILON_START,
                "epsilon end": EPSILON_END,
                "epsilon decay step": EPSILON_DECAY_STEP, 
                "target update freq": TARGET_UPDATE_FREQ, 
                "learning rate": LEARNING_RATE,
                "total step": TOTAL_STEP,
                })

    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = gym.make('CartPole-v0')
    env.seed(seed)
    rew_buffer = deque([0.0], maxlen=100)
    length_buffer = deque([0.0], maxlen=100)
    episode_reward = 0.0
    episode_length = 0.0

    agent = DQN(input_dims=int(np.prod(env.observation_space.shape)), n_actions=env.action_space.n)
    agent.update_target_network()

    obs = env.reset()
    for _ in range(MIN_REPLAY_SIZE):
        action = env.action_space.sample()
        new_obs, rew, done, info = env.step(action)
        transition = (obs, action, rew, done, new_obs)
        agent.store_transition(transition=transition)
        obs = new_obs
        if done:
            obs = env.reset()

    # Main Training Loop
    obs = env.reset()
    for step in range(TOTAL_STEP):
        action = agent.act(obs)
        new_obs, rew, done, info = env.step(action)
        transition = (obs, action, rew, done, new_obs)
        agent.store_transition(transition=transition)
        obs = new_obs
        episode_reward += rew
        if done:
            obs = env.reset()
            rew_buffer.append(episode_reward)
            episode_reward = 0.0

        # After solved, watch it play
        if len(rew_buffer) >= 100:
            if np.mean(rew_buffer) >= 195:
                while True:
                    action = agent.act(obs)
                    obs, _, done, _ = env.step(action)
                    env.render()
                    if done:
                        env.reset()

        agent.learn()
        if step % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        # Logging
        wandb.log({'Avg Reward': np.mean(rew_buffer), 'epsilon': agent.epsilon})
        if step % 1000 == 0:
            print()
            print('Step', step)
            print('Avg Rew', np.mean(rew_buffer))
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from collections import deque
import itertools
import random

GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = 50_000
MIN_REPLAY_SIZE = 1_000 
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10_000 
TARGET_UPDATE_FREQ = 1_000 
LEARNING_RATE = 5e-4

class DeepQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()

        in_features = int(np.prod(env.observation_space.shape))

        self.model = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, env.action_space.n),
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

    def store_transition(self, transition):
        self.replay_buffer.append(transition)

    def generate_batch(self):
        batch = random.sample(self.replay_buffer, BATCH_SIZE)
        
        return batch

class DQN:
    def __init__(self):
        pass
    
if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    
    # replay_buffer = deque(maxlen=BUFFER_SIZE)
    replay_buffer = ReplayBuffer(buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE)
    rew_buffer = deque([0.0], maxlen=100)

    episode_reward = 0.0

    online_net = DeepQNetwork(env)
    target_net = DeepQNetwork(env)

    target_net.load_state_dict(online_net.state_dict())

    optimizer = torch.optim.Adam(online_net.parameters(), lr=LEARNING_RATE)

    # Initialize Replay Buffer
    obs = env.reset()

    for _ in range(MIN_REPLAY_SIZE):
        action = env.action_space.sample()

        new_obs, rew, done, info = env.step(action)
        transition = (obs, action, rew, done, new_obs)
        # replay_buffer.append(transition)
        replay_buffer.store_transition(transition=transition)

        obs = new_obs

        if done:
            obs = env.reset()

    # Main Training Loop
    obs = env.reset()
    for step in itertools.count():
        epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
        
        rnd_sample = random.random()
        
        if rnd_sample <= epsilon:
            action = env.action_space.sample()
        else:
            action = online_net.act(obs)

        new_obs, rew, done, info = env.step(action)
        transition = (obs, action, rew, done, new_obs)
        # replay_buffer.append(transition)
        replay_buffer.store_transition(transition=transition)
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
                    action = online_net.act(obs)

                    obs, _, done, _ = env.step(action)
                    env.render()
                    if done:
                        env.reset()

        # Start Gradient Step
        # transitions = random.sample(replay_buffer, BATCH_SIZE)
        batch = replay_buffer.generate_batch()

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
        target_q_values = target_net(new_obses_t)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

        targets = rews_t + GAMMA * (1 - dones_t) * max_target_q_values

        # Compute Loss
        q_values = online_net(obses_t)

        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

        loss = nn.functional.smooth_l1_loss(action_q_values, targets)

        # Gradient Descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update Target Network
        if step % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(online_net.state_dict())

        # Logging
        if step % 1000 == 0:
            print()
            print('Step', step)
            print('Avg Rew', np.mean(rew_buffer))
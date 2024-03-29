import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import gym
import wandb
import random
#TODO: debug.

class DeepQNetwork(nn.Module):
    """
        2 layers deep neural network used for estimating q values.
    """
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model = nn.Sequential(
            nn.Linear(self.input_dims, self.fc1_dims),
            nn.ReLU(),
            nn.Linear(self.fc1_dims, self.fc2_dims),
            nn.ReLU(),
            nn.Linear(self.fc2_dims, self.n_actions),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        q = self.model(state)

        return q

class ReplayBuffer:
    """
        replaybuffer is used to store the transitions, and then generating batch for training
    """
    def __init__(self, input_dims, batch_size, mem_size=100_000):
        self.batch_size = batch_size
        self.mem_size = mem_size
        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.new_state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.done_memory = np.zeros(self.mem_size, dtype=np.bool8)
        self.mem_cntr = 0
    
    def store_transition(self, state, action, new_state, reward, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        self.done_memory[index] = done

        self.mem_cntr += 1
    
    def generate_batch(self, device):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(device)
        action_batch = self.action_memory[batch]
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(device)
        done_batch = T.tensor(self.done_memory[batch]).to(device)

        return batch_index, state_batch, action_batch, new_state_batch, reward_batch, done_batch


class DQN():
    """
        DQN algorithm with replaybuffer.(without targetnetwork now)
    """
    def __init__(self, gamma, epsilon, lr, input_dims, n_actions, batch_size,
                 max_mem_size=100_000, eps_min=0.01, eps_dec=5e-4, replace_target_cnt=1):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0
        self.replace_target_cnt = replace_target_cnt
        self.Q_eval = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims,
                                   fc1_dims=256, fc2_dims=256)
        self.Q_target = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims,
                                     fc1_dims=256, fc2_dims=256)
        self.update_target_network()
        self.replaybuffer = ReplayBuffer(input_dims=input_dims, batch_size=batch_size, mem_size=max_mem_size)
        
    def store_transition(self, state, action, new_state, reward, done):
        self.replaybuffer.store_transition(state=state, action=action, new_state=new_state, reward=reward, done=done)

    def choose_action(self, obs):
        if np.random.random() > self.epsilon:
            state_t = T.tensor(obs).unsqueeze(0).to(self.Q_eval.device)
            q = self.Q_eval.forward(state_t)
            action = T.argmax(q).detach().item()
        else:
            action = np.random.choice(self.action_space)

        return action
    
    def update_target_network(self):
        self.Q_target.load_state_dict(self.Q_eval.state_dict())
    
    def learn(self):
        if self.replaybuffer.mem_cntr < self.replaybuffer.batch_size:
            return
        
        self.Q_eval.optimizer.zero_grad()
        
        batch_index, state_batch, action_batch, new_state_batch, reward_batch, done_batch = self.replaybuffer.generate_batch(self.Q_eval.device)

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        #NOTE: when use Q_target, its performance is low, why?
        q_next = self.Q_eval.forward(new_state_batch)
        q_next_ = self.Q_target.forward(new_state_batch)
        q_next[done_batch] = 0.0
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.learn_step_counter += 1
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.update_target_network()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
                        else self.epsilon
        
if __name__ == '__main__':
    seed = 42
    random.seed(seed)
    T.manual_seed(seed)
    np.random.seed(seed)
    env = gym.make('LunarLander-v2')
    env.seed(seed)
    env_eval = gym.make('LunarLander-v2')
    env_eval.seed(seed)

    wandb.init(project="classic-drl-algorithms", entity="cc299792458", name='dqn_wo_tgtn')

    agent = DQN(gamma=0.99,
                epsilon=1.0, 
                lr=0.003, 
                input_dims=env.observation_space.shape[0],
                n_actions=env.action_space.n,
                batch_size=64,
                eps_min=0.01)
    scores, eps_history = [], []
    n_games = 1500

    for i in range(n_games):
        score = 0
        done = False
        obs = env.reset()
        while not done:
            # rollout is a single worker.
            action = agent.choose_action(obs)
            new_obs, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(state=obs, 
                                   action=action, 
                                   new_state=new_obs,
                                   reward=reward,
                                   done=done)
            agent.learn()
            obs = new_obs
        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])
        
        print('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score, 'epsilon %.2f' % agent.epsilon)
        wandb.log({'score': score, 'avg_score': avg_score, 'epsilon': agent.epsilon})

        # Render to see the performance of trained agent.
        if i % 50 == 0:
            obs = env_eval.reset()
            done = False
            while not done:
                env_eval.render()
                action = agent.choose_action(obs)
                obs, reward, done, info = env_eval.step(action)
            # env.close()
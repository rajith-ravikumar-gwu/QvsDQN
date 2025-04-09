import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
from time import time

# Environment setup
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Hyperparameters
LEARNING_RATE = 0.001
GAMMA = 0.95
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
MEMORY_SIZE = 10000
EPISODES = 500

# Q-Learning Implementation
class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((10, 10, 10, 10, action_size))  # Discretized state space
        self.epsilon = EPSILON
        
    def discretize_state(self, state):
        # Discretize continuous state space into bins
        state_bounds = list(zip([-4.8, -4, -0.418, -4], [4.8, 4, 0.418, 4]))
        state_bins = [10] * len(state)
        discrete_state = []
        for i, (s, bounds) in enumerate(zip(state, state_bounds)):
            discrete_state.append(int(np.digitize(s, np.linspace(bounds[0], bounds[1], state_bins[i])) - 1))
        return tuple(discrete_state)
    
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state = self.discretize_state(state)
        return np.argmax(self.q_table[state])
    
    def train(self, state, action, reward, next_state, done):
        state = self.discretize_state(state)
        next_state = self.discretize_state(next_state)
        
        if done:
            target = reward
        else:
            target = reward + GAMMA * np.max(self.q_table[next_state])
        
        self.q_table[state][action] = (1 - LEARNING_RATE) * self.q_table[state][action] + LEARNING_RATE * target
        
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

# DQN Implementation
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values).item()
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([i[0] for i in minibatch])
        actions = torch.LongTensor([i[1] for i in minibatch])
        rewards = torch.FloatTensor([i[2] for i in minibatch])
        next_states = torch.FloatTensor([i[3] for i in minibatch])
        dones = torch.FloatTensor([i[4] for i in minibatch])
        
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Benchmarking function
def benchmark_agent(agent_class, agent_name):
    agent = agent_class(state_size, action_size)
    scores = []
    times = []
    solved_episode = None
    
    for episode in range(EPISODES):
        state, _ = env.reset()
        score = 0
        start_time = time()
        
        while True:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            if isinstance(agent, DQNAgent):
                agent.remember(state, action, reward, next_state, done)
                agent.replay(BATCH_SIZE)
            else:
                agent.train(state, action, reward, next_state, done)
            
            state = next_state
            score += reward
            
            if done:
                break
        
        end_time = time()
        episode_time = end_time - start_time
        scores.append(score)
        times.append(episode_time)
        
        # Check if environment is solved (average score over last 100 episodes >= 195)
        if len(scores) >= 100:
            avg_score = np.mean(scores[-100:])
            if avg_score >= 195 and solved_episode is None:
                solved_episode = episode
                print(f"\n{agent_name} solved the environment at episode {episode + 1}!")
                print(f"Average score over last 100 episodes: {avg_score:.2f}")
        
        if (episode + 1) % 10 == 0:
            avg_score = np.mean(scores[-10:])
            print(f"{agent_name} - Episode: {episode + 1}, Average Score: {avg_score:.2f}, Time: {episode_time:.2f}s")
    
    if solved_episode is None:
        print(f"\n{agent_name} did not solve the environment within {EPISODES} episodes.")
        print(f"Best average score over 100 episodes: {np.max([np.mean(scores[i:i+100]) for i in range(len(scores)-99)]) if len(scores) >= 100 else np.mean(scores):.2f}")
    
    return scores, times, solved_episode

# Run benchmarks
print("Starting Q-Learning benchmark...")
ql_scores, ql_times, ql_solved = benchmark_agent(QLearningAgent, "Q-Learning")

print("\nStarting DQN benchmark...")
dqn_scores, dqn_times, dqn_solved = benchmark_agent(DQNAgent, "DQN")

# Plot results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(ql_scores, label='Q-Learning')
plt.plot(dqn_scores, label='DQN')
if ql_solved is not None:
    plt.axvline(x=ql_solved, color='blue', linestyle='--', label='Q-Learning Solved')
if dqn_solved is not None:
    plt.axvline(x=dqn_solved, color='orange', linestyle='--', label='DQN Solved')
plt.title('Scores over Episodes')
plt.xlabel('Episode')
plt.ylabel('Score')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(ql_times, label='Q-Learning')
plt.plot(dqn_times, label='DQN')
plt.title('Time per Episode')
plt.xlabel('Episode')
plt.ylabel('Time (s)')
plt.legend()

plt.tight_layout()
plt.savefig('benchmark_results.png')
plt.show()

# Print final statistics
print("\nFinal Statistics:")
print(f"Q-Learning - Average Score: {np.mean(ql_scores):.2f}, Average Time: {np.mean(ql_times):.2f}s")
print(f"DQN - Average Score: {np.mean(dqn_scores):.2f}, Average Time: {np.mean(dqn_times):.2f}s")    
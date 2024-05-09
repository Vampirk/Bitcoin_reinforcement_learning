import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque, namedtuple
import numpy as np
from environment import FuturesTradingEnv
from config import GAMMA, BATCH_SIZE, BUFFER_SIZE, LEARNING_RATE
from config import UPDATE_EVERY, NEURAL_NET_CONFIG, SEED, NUM_EPISODES, MAX_EPISODE_LEN

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# 경험 저장을 위한 Named Tuple 정의
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

# 레이어 정규화 포함된 신경망 정의
class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.ln1 = nn.LayerNorm(hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.ln2 = nn.LayerNorm(hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)

# 듀얼링 DQN 아키텍처 정의
class DuelingQNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.ln1 = nn.LayerNorm(hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.ln2 = nn.LayerNorm(hidden_sizes[1])
        
        self.value_stream = nn.Linear(hidden_sizes[1], 1)
        self.advantage_stream = nn.Linear(hidden_sizes[1], output_size)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values

# DQN 에이전트 정의
class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.input_size = NEURAL_NET_CONFIG['input_size']
        self.hidden_sizes = NEURAL_NET_CONFIG['hidden_sizes']
        self.output_size = NEURAL_NET_CONFIG['output_size']
        
        # Q-Network 및 타깃 Q-Network 초기화 (듀얼링 DQN 사용)
        self.q_network = DuelingQNetwork(self.input_size, self.hidden_sizes, self.output_size)
        self.target_q_network = DuelingQNetwork(self.input_size, self.hidden_sizes, self.output_size)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        # 옵티마이저 및 손실 함수 정의
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.SmoothL1Loss()  # Huber Loss
        
        # 리플레이 버퍼 초기화
        self.replay_buffer = deque(maxlen=BUFFER_SIZE)
        self.priority_weights = deque(maxlen=BUFFER_SIZE)
        
        # 에피소드 및 타깃 업데이트 카운터 초기화
        self.episode_count = 0
        self.target_update_count = 0

    def get_action(self, state, epsilon):
        # 입실론-탐욕 정책으로 행동 선택
        if random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            q_values = self.q_network(state_tensor)
            action = torch.argmax(q_values).item()
        return action

    def update_replay_buffer(self, state, action, reward, next_state, done):
        # 리플레이 버퍼에 경험 저장 (우선순위 없음)
        experience = Experience(state, action, reward, next_state, done)
        self.replay_buffer.append(experience)
        self.priority_weights.append(1.0)  # 초기 우선순위는 모두 1로 설정

    def update_q_network(self):
        # 우선순위에 따라 배치 샘플링
        priorities = self.priority_weights
        priorities = np.array(priorities) ** (1 / 0.6)  # 우선순위 가중치 정규화
        priorities /= priorities.sum()
        indices = np.random.choice(len(self.replay_buffer), BATCH_SIZE, p=priorities)
        minibatch = [self.replay_buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = map(np.array, zip(*minibatch))
        
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)
        
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q_values = self.target_q_network(next_states).max(1)[0]
        next_q_values[dones] = 0.0  # 종료 상태에서는 0으로 설정
        
        # 더블 DQN 알고리즘 적용
        q_values_next = self.q_network(next_states)
        best_actions = torch.argmax(q_values_next, dim=1)
        next_q_values_double = self.target_q_network(next_states).gather(1, best_actions.unsqueeze(1)).squeeze()
        
        # 미래 보상을 고려하여 최종 보상 계산
        expected_q_values = rewards + GAMMA * next_q_values_double
        
        loss = self.loss_fn(q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.target_update_count += 1
        if self.target_update_count % UPDATE_EVERY == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

    def train(self, num_episodes, max_episode_len):
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
        
            for step in range(max_episode_len):
                epsilon = max(0.01, 0.9 - 0.01 * (episode / 200))
                action = self.get_action(state, epsilon)
                next_state, reward, done, _ = self.env.step(action)
                self.update_replay_buffer(state, action, reward, next_state, done)
                episode_reward += reward
                state = next_state

                if len(self.replay_buffer) >= BATCH_SIZE:
                    self.update_q_network()

                if done:
                    break

            self.episode_count += 1
            print(f"Episode {self.episode_count}: Reward = {episode_reward} Account = {env.account_balance}")

# DQN 에이전트 초기화 및 학습 진행
env = FuturesTradingEnv()  # 환경 객체 생성
agent = DQNAgent(env)
agent.train(NUM_EPISODES, MAX_EPISODE_LEN)


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os
import time

# Check if CUDA is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the game environment
class MiniGameEnv:
    def __init__(self, n):
        self.n = n
        self.reset()

    def reset(self):
        self.player1 = [1, 1]
        self.player2 = [1, 1]
        self.current_player = 1
        return self.get_state()

    def get_state(self):
        # Return raw numbers for display
        return (tuple(self.player1), tuple(self.player2), self.current_player)

    def get_normalized_state(self):
        # Return normalized numbers for neural network
        p1_normalized = [x/self.n for x in self.player1]
        p2_normalized = [x/self.n for x in self.player2]
        return (tuple(p1_normalized), tuple(p2_normalized), self.current_player)

    def step(self, action):
        if self.current_player == 1:
            # Player 1's turn: Add P2's number to P1's chosen number
            own_idx, opp_idx = divmod(action, 2)
            add_value = self.player2[opp_idx]
            self.player1[own_idx] += add_value
        else:
            # Player 2's turn: Add P1's number to P2's chosen number
            # For Player 2, we flip the indices since their state shows [P2, P1]
            opp_idx, own_idx = divmod(action, 2)  # Flip the interpretation
            add_value = self.player1[opp_idx]
            self.player2[own_idx] += add_value

        # Calculate current sums
        p1_sum = sum(self.player1)
        p2_sum = sum(self.player2)

        # Check win condition from current player's perspective
        if self.current_player == 1:
            if p1_sum >= self.n:
                return self.get_state(), 25, True  # P1 wins
            if p2_sum >= self.n:
                return self.get_state(), -25, True  # P2 wins
        else:
            if p2_sum >= self.n:
                return self.get_state(), 25, True  # P2 wins
            if p1_sum >= self.n:
                return self.get_state(), -25, True  # P1 wins

        # Switch player
        self.current_player = 3 - self.current_player
        return self.get_state(), 0, False  # No reward during game, only at end

def print_state(state):
    # Get the raw numbers from the environment instead of normalized state
    p1_numbers = [int(x) for x in state[0]]  # Convert to integers
    p2_numbers = [int(x) for x in state[1]]  # Convert to integers
    current_player = state[2]
    print(f"\nPlayer 1 numbers: [{p1_numbers[0]}, {p1_numbers[1]}]")
    print(f"Player 2 numbers: [{p2_numbers[0]}, {p2_numbers[1]}]")
    print(f"Current player: {current_player}")

def play_example_game(env, agent1, agent2, verbose=True):
    state = env.reset()
    if verbose:
        print("\nNew game starting!")
        print_state(state)
    
    game_log = []
    done = False
    total_moves = 0
    
    while not done and total_moves < 50:  # Add move limit to prevent infinite games
        total_moves += 1
        current_agent = agent1 if state[2] == 1 else agent2
        
        # Get normalized state for neural network
        norm_state = env.get_normalized_state()
        player_encoding = [1.0, 0.0] if state[2] == 1 else [0.0, 1.0]
        
        # Arrange state from current player's perspective
        if state[2] == 1:  # Player 1's turn
            nn_state = np.array(norm_state[0] + norm_state[1] + tuple(player_encoding))
        else:  # Player 2's turn
            nn_state = np.array(norm_state[1] + norm_state[0] + tuple(player_encoding))
        
        # Get action
        action = current_agent.act(nn_state)
        own_idx, opp_idx = divmod(action, 2)
        
        # Log the move
        if verbose:
            player = "Player 1" if state[2] == 1 else "Player 2"
            print(f"\n{player}'s turn...")
            print(f"{player} adds opponent's number {opp_idx} to their number {own_idx}")
        
        # Make move
        next_state, reward, done = env.step(action)
        if verbose:
            print_state(next_state)
        
        if done:
            if verbose:
                winner = "Player 1" if reward == 25 and state[2] == 1 or reward == -25 and state[2] == 2 else "Player 2"
                print(f"\nGame Over! {winner} wins!")
        
        state = next_state
    
    return total_moves

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(50000)
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.3
        self.learning_rate = 0.0005
        self.model = DQN(state_size, action_size).to(device)
        self.target_model = DQN(state_size, action_size).to(device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.temperature = 6
        self.target_update_counter = 0
        self.target_update_freq = 10

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(device)
        
        with torch.no_grad():
            q_values = self.model(state)
            # Add temperature scaling for more exploration
            q_values = q_values / self.temperature
            # Use softmax instead of argmax for more diverse actions
            probs = torch.softmax(q_values, dim=0)
            action = np.random.choice(self.action_size, p=probs.cpu().numpy())
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = self.memory.sample(batch_size)
        states = torch.FloatTensor(np.vstack([x[0] for x in minibatch])).to(device)
        actions = torch.LongTensor(np.array([x[1] for x in minibatch])).to(device)
        rewards = torch.FloatTensor(np.array([x[2] for x in minibatch])).to(device)
        next_states = torch.FloatTensor(np.vstack([x[3] for x in minibatch])).to(device)
        dones = torch.FloatTensor(np.array([x[4] for x in minibatch])).to(device)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
        self.model.eval()

# Training loop
if __name__ == "__main__":
    n = 10
    env = MiniGameEnv(n)
    state_size = 6
    action_size = 4
    agent1 = DQNAgent(state_size, action_size)
    agent2 = DQNAgent(state_size, action_size)

    episodes = 5000
    batch_size = 128

    # Keep track of scores for monitoring
    scores = []
    epsilon_history = []
    
    print("\nPlaying one example game before training to see initial behavior:")
    play_example_game(env, agent1, agent2)
    
    print("\nStarting training...")
    for e in range(episodes):
        state = env.reset()
        # Alternate starting player
        if e % 2 == 0:
            env.current_player = 1  # Agent1 goes first
        else:
            env.current_player = 2  # Agent2 goes first
            
        game_over = False
        moves = 0
        final_reward = 0  # Track the final reward
        episode_start_time = time.time()  # Start timing
        
        while not game_over and moves < 50:  # Add move limit
            moves += 1
            current_agent = agent1 if state[2] == 1 else agent2
            
            # Get normalized state for neural network
            norm_state = env.get_normalized_state()
            player_encoding = [1.0, 0.0] if state[2] == 1 else [0.0, 1.0]
            
            # Arrange state from current player's perspective
            if state[2] == 1:  # Player 1's turn
                nn_state = np.array(norm_state[0] + norm_state[1] + tuple(player_encoding))
            else:  # Player 2's turn
                nn_state = np.array(norm_state[1] + norm_state[0] + tuple(player_encoding))
            
            # Get action
            action = current_agent.act(nn_state)
            next_state, reward, game_over = env.step(action)
            
            if game_over:
                final_reward = reward  # No need to flip, already from player's perspective
            
            # Get normalized next state for neural network
            next_norm_state = env.get_normalized_state()
            next_player_encoding = [1.0, 0.0] if next_state[2] == 1 else [0.0, 1.0]
            
            # Arrange next state from current player's perspective
            if next_state[2] == 1:  # Player 1's turn
                next_nn_state = np.array(next_norm_state[0] + next_norm_state[1] + tuple(next_player_encoding))
            else:  # Player 2's turn
                next_nn_state = np.array(next_norm_state[1] + next_norm_state[0] + tuple(next_player_encoding))
            
            # Store experience in memory (reward already from player's perspective)
            current_agent.remember(nn_state, action, reward, next_nn_state, game_over)
            
            state = next_state
            
            # Train on a batch of experiences
            if len(current_agent.memory) > batch_size:
                current_agent.replay(batch_size)
        
        # Update exploration rate
        agent1.epsilon = max(agent1.epsilon_min, agent1.epsilon * agent1.epsilon_decay)
        agent2.epsilon = max(agent2.epsilon_min, agent2.epsilon * agent2.epsilon_decay)
        
        # Log every episode
        episode_time = (time.time() - episode_start_time) * 1000  # Convert to milliseconds
        print(f"Episode: {e}/{episodes}, Player 1 Score: {final_reward:+d}, Epsilon: {agent1.epsilon:.3f}, Moves: {moves}, Time: {episode_time:.0f}ms")
        
        # Detailed example game every 1000 episodes
        if e > 0 and e % 250 == 0:
            print("\nPlaying example game to show current performance:")
            moves = play_example_game(env, agent1, agent2)
            print(f"Example game completed in {moves} moves")

    print("Training completed!")
    # Save the trained models
    models_dir = "trained_models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    agent1.save_model(os.path.join(models_dir, 'agent1_model.pth'))
    agent2.save_model(os.path.join(models_dir, 'agent2_model.pth'))
    print(f"Models saved in {models_dir} directory")
import torch
import numpy as np
from main import MiniGameEnv, DQNAgent, device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def print_state(state):
    p1_numbers = [int(x) for x in state[0]]  # Convert to integers
    p2_numbers = [int(x) for x in state[1]]
    print(f"\nPlayer 1 numbers: {p1_numbers}")
    print(f"Player 2 numbers: {p2_numbers}")

def get_player_move():
    while True:
        try:
            print("\nYour move!")
            print("Choose which of your numbers to add to (0 or 1):")
            own_idx = int(input())
            print("Choose which opponent number to add (0 or 1):")
            opp_idx = int(input())
            
            if own_idx in [0, 1] and opp_idx in [0, 1]:
                # Player 2's action encoding is flipped: opp_idx * 2 + own_idx
                return opp_idx * 2 + own_idx
            else:
                print("Invalid input! Please enter 0 or 1")
        except ValueError:
            print("Invalid input! Please enter numbers")

def play_game():
    n = 10  # Same as training
    env = MiniGameEnv(n)
    state_size = 6  # Match the trained model size
    action_size = 4
    
    # Load the trained agent
    agent = DQNAgent(state_size, action_size)
    try:
        agent.load_model('trained_models/agent1_model.pth')
        print("Loaded trained model successfully!")
    except Exception as e:
        print("Could not load model! Make sure you have trained the model first.")
        print("Error:", e)
        return
    
    # Set agent to evaluation mode (no random moves and very deterministic)
    agent.epsilon = 0
    agent.temperature = 0.001  # Very low temperature for more deterministic behavior
    
    while True:
        state = env.reset()
        game_over = False
        
        print("\nNew game starting!")
        print("You are Player 2, AI is Player 1")
        print("Goal: Get your sum to reach or exceed", n)
        
        while not game_over:
            print_state(state)
            
            if env.current_player == 1:
                # AI's turn
                print("\nAI's turn...")
                # Get normalized state for neural network
                norm_state = env.get_normalized_state()
                player_encoding = [1.0, 0.0]  # Player 1 encoding
                # Agent1 sees own numbers first
                nn_state = np.array(norm_state[0] + norm_state[1] + tuple(player_encoding))
                action = agent.act(nn_state)
                own_idx, opp_idx = divmod(action, 2)  # Player 1's action interpretation
                print(f"AI adds your number {opp_idx} to its number {own_idx}")
            else:
                # Player's turn
                action = get_player_move()
            
            # Make the move and get new state
            state, reward, game_over = env.step(action)
            
            if game_over:
                print_state(state)
                p1_sum = sum(state[0])
                p2_sum = sum(state[1])
                if p2_sum >= n:  # You won (Player 2)
                    print("\nYou won!")
                else:  # AI won (Player 1)
                    print("\nAI won!")
        
        play_again = input("\nPlay again? (y/n): ").lower()
        if play_again != 'y':
            break

if __name__ == "__main__":
    play_game()

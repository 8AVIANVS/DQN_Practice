import torch
import numpy as np
import random
from main import MiniGameEnv, DQNAgent, device
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def random_move(state, action_size=4):
    """Bot player that makes random moves"""
    return random.randrange(action_size)

def print_state(state):
    print(f"\nPlayer 1 numbers: {state[0]}")
    print(f"Player 2 numbers: {state[1]}")

def benchmark_agent(num_games):
    n = 10
    env = MiniGameEnv(n)
    state_size = 6
    action_size = 4

    # Load the trained agent (Player 1)
    agent = DQNAgent(state_size, action_size)
    try:
        agent.load_model('trained_models/agent1_model.pth')
        print("Loaded trained model successfully!")
    except Exception as e:
        print("Could not load model! Make sure you have trained the model first.")
        print("Error:", e)
        return

    # Set agent to evaluation mode
    agent.epsilon = 0
    agent.temperature = 0.001

    wins = 0
    total_moves = 0
    game_logs = []

    for game in range(1, num_games + 1):
        state = env.reset()
        moves = 0
        done = False
        
        # Determine if this is a game to show detailed steps
        show_details = (game % 100 == 0)
        
        if show_details:
            print("\n" + "="*50)
            print(f"Detailed Game #{game}")
            print("="*50)
            print("\nNew game starting!")
            print_state(state)

        while not done and moves < 50:  # Move limit to prevent infinite games
            moves += 1
            current_player = state[2]

            if current_player == 1:  # Agent's turn
                # Get normalized state for neural network
                norm_state = env.get_normalized_state()
                player_encoding = [1.0, 0.0]  # Player 1 encoding
                # Agent1 sees own numbers first
                nn_state = np.array(norm_state[0] + norm_state[1] + tuple(player_encoding))
                
                # Get agent's action
                action = agent.act(nn_state)
                if show_details:
                    own_idx, opp_idx = divmod(action, 2)
                    print(f"\nAgent's turn...")
                    print(f"Agent adds opponent's number {opp_idx} to their number {own_idx}")
            else:  # Random bot's turn
                action = random_move(state)
                if show_details:
                    own_idx, opp_idx = divmod(action, 2)
                    print(f"\nRandom Bot's turn...")
                    print(f"Bot adds opponent's number {opp_idx} to their number {own_idx}")

            # Make move
            state, reward, done = env.step(action)
            if show_details:
                print_state(state)

            if done:
                # Agent (Player 1) wins if their sum reaches target
                p1_sum = sum(state[0])
                if p1_sum >= n:
                    wins += 1
                    if show_details:
                        print("\nGame Over! Agent wins!")
                else:
                    if show_details:
                        print("\nGame Over! Random Bot wins!")
                total_moves += moves
                game_logs.append(moves)

        # Print progress every 100 games
        if game % 100 == 0:
            win_rate = (wins / game) * 100
            avg_moves = sum(game_logs[-100:]) / len(game_logs[-100:])
            print("\nProgress Statistics:")
            print(f"Games played: {game}/{num_games}")
            print(f"Win rate: {win_rate:.2f}%")
            print(f"Average moves in last 100 games: {avg_moves:.2f}")
            print("-" * 50)

    # Final statistics
    final_win_rate = (wins / num_games) * 100
    avg_moves_total = total_moves / num_games
    print("\nFinal Statistics:")
    print(f"Total games played: {num_games}")
    print(f"Final win rate: {final_win_rate:.2f}%")
    print(f"Average moves per game: {avg_moves_total:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark trained agent against random bot')
    parser.add_argument('--games', type=int, default=1000, help='Number of games to play')
    args = parser.parse_args()
    
    benchmark_agent(args.games)

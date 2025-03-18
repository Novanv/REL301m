import numpy as np
import pickle
from xo_env import TicTacToeEnv

class QLearningAgent:
    """
    Q-Learning Agent for Tic-Tac-Toe game.
    
    Attributes:
        q_table (dict): Stores Q-values for each state.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        epsilon (float): Exploration rate.
        epsilon_decay (float): Decay rate for epsilon.
        list_q_value (list): Stores Q-values for analysis.
    """
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.9, epsilon_decay=0.000001):
        self.q_table = {}  # Q-table stored as a dictionary
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay  # Epsilon decay rate
        self.list_q_value = []

    def get_q_values(self, state):
        """Retrieve Q-values for a given state."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(9)  # Each state has 9 possible actions
        return self.q_table[state]

    def choose_action(self, state):
        """Choose an action using epsilon-greedy policy."""
        available_actions = [i for i in range(9) if state[i] == 0]  # Select empty positions

        if np.random.rand() < self.epsilon:
            return np.random.choice(available_actions)  # Choose randomly

        q_values = self.get_q_values(state)
        return max(available_actions, key=lambda x: q_values[x])  # Choose action with max Q-value

    def update(self, state, action, reward, next_state):
        """Update the Q-value using the Q-learning update rule."""
        q_values = self.get_q_values(state)
        next_q_values = self.get_q_values(next_state)
        q_values[action] += self.alpha * (reward + self.gamma * np.max(next_q_values) - q_values[action])

    def save_model(self, filename="q_table.pkl"):
        """Save Q-table to a file."""
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)

    def load_model(self, filename="q_table.pkl"):
        """Load Q-table from a file."""
        with open(filename, "rb") as f:
            self.q_table = pickle.load(f)

    def save_q_table_txt(self, filename="q_table.txt"):
        """Save Q-table to a text file for inspection."""
        with open(filename, "w") as f:
            for state, q_values in self.q_table.items():
                f.write(f"State: {list(state)}\n")
                f.write(f"Q_value: {q_values.tolist()}\n")
                f.write("-" * 50 + "\n")
        print(f"Q-table saved to {filename}")

    def train_agent(self, episodes=1000000):
        """Train the Q-learning agent by playing against a random player."""
        env = TicTacToeEnv()

        for episode in range(episodes):
            state = env.reset()
            done = False

            while not done:
                # Random player (Player 2) chooses an action randomly
                available_actions = [i for i in range(9) if state[i] == 0]
                if available_actions:
                    player_action = np.random.choice(available_actions)
                    state, _, done = env.step(player_action, 2)  # Player 2 moves first

                if done:  # Stop if player wins or draws
                    break

                # Q-learning Agent (Player 1) selects action based on Q-values
                action = self.choose_action(state)
                next_state, reward, done = env.step(action, 1)  # Agent moves as Player 1
                self.update(state, action, reward, next_state)
                state = next_state
            
            self.epsilon -= self.epsilon_decay  # Decrease epsilon for exploration-exploitation tradeoff

        self.save_model()
        self.save_q_table_txt()
        print("Training Completed!")

if __name__ == "__main__":
    agent = QLearningAgent()
    agent.train_agent()

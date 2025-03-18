import numpy as np

class TicTacToeEnv:
    """
    Tic-Tac-Toe environment for training a Q-learning agent.
    
    Attributes:
        state (np.array): 3x3 board representing the game state.
        done (bool): Indicates if the game is over.
        winner (int or None): Stores the winner (1 for agent, 2 for player, None if no winner).
    """
    def __init__(self):
        """Initialize the Tic-Tac-Toe environment."""
        self.state = np.zeros((3, 3), dtype=int)  # 3x3 Tic-Tac-Toe board
        self.done = False  # Flag to indicate game over
        self.winner = None  # Stores the winner

    def reset(self):
        """Reset the game to the initial state."""
        self.state.fill(0)
        self.done = False
        self.winner = None
        return self.get_state()

    def get_state(self):
        """Convert the board state to a tuple for Q-table usage."""
        return tuple(self.state.flatten())  # Convert matrix to a 1D vector

    def check_winner(self):
        """Check for a winner in the game."""
        for i in range(3):
            # Check rows
            if np.all(self.state[i, :] == 1):
                return 1  # Agent (X) wins
            if np.all(self.state[i, :] == 2):
                return 2  # Player (O) wins

            # Check columns
            if np.all(self.state[:, i] == 1):
                return 1
            if np.all(self.state[:, i] == 2):
                return 2

        # Check main diagonal
        if np.all(np.diag(self.state) == 1):
            return 1
        if np.all(np.diag(self.state) == 2):
            return 2

        # Check anti-diagonal
        if np.all(np.diag(np.fliplr(self.state)) == 1):
            return 1
        if np.all(np.diag(np.fliplr(self.state)) == 2):
            return 2

        return None  # No winner yet

    def step(self, action, player):
        """
        Execute a move in the game.
        
        Args:
            action (int): The index (0-8) where the player wants to move.
            player (int): The player making the move (1 for agent, 2 for human player).
        
        Returns:
            tuple: (new state, reward, done flag)
        """
        if self.state[action // 3, action % 3] != 0 or self.done:
            return self.get_state(), -10, True  # Invalid action (penalty)

        # Place the player's move on the board
        self.state[action // 3, action % 3] = player

        # Check if the move resulted in a win
        winner = self.check_winner()

        if winner is not None:
            self.done = True
            self.winner = winner
            return self.get_state(), 5 if winner == 1 else -5, True  # Reward for win/loss

        # Check for a draw (no empty spaces left)
        if not (self.state == 0).any():
            self.done = True
            return self.get_state(), 0, True  # Draw, no reward

        return self.get_state(), 0, False  # Continue playing

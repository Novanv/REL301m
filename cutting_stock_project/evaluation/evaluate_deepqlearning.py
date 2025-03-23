import gymnasium as gym
import numpy as np
import torch
from env.CuttingStockEnvOptimized import CuttingStockEnvOptimized
from agents.deep_qlearning_agent import DeepQLearningAgent, state_to_tensor
import sys
import os

def evaluate_agent(checkpoint_path, num_episodes=10):
    """
    Evaluate the Deep Q-Learning agent using a saved checkpoint.

    This function:
      - Initializes the CuttingStockEnvOptimized environment.
      - Creates a DeepQLearningAgent.
      - Loads the policy network weights from the checkpoint.
      - Sets epsilon to 0 to force greedy action selection.
      - Runs the agent for a specified number of episodes, printing the total reward per episode and the average reward.

    Args:
        checkpoint_path (str): Path to the saved checkpoint file.
        num_episodes (int): Number of episodes to run for evaluation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use human render_mode to see the visual interface; change to "rgb_array" if needed.
    env = CuttingStockEnvOptimized(render_mode="human")
    
    agent = DeepQLearningAgent(env, device)
    
    if not os.path.exists(checkpoint_path):
        print(f"checkpoints/Case_new'{checkpoint_path}' does not exist.")
        sys.exit(1)
        
    try:
        agent.policy_net.load_state_dict(torch.load(checkpoint_path, map_location=device))
        agent.policy_net.eval()  # Set the network to evaluation mode.
        print(f"Loaded checkpoint from '{checkpoint_path}'")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)
    
    # Set epsilon to zero so that the agent always selects the best action.
    agent.epsilon = 0.0
    total_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state_tensor = state_to_tensor(state, device)
        episode_reward = 0
        done = False
        while not done:
            # Select action using the greedy policy.
            action_index = agent.select_action(state_tensor)
            
            # Convert discrete action index back to the dictionary format required by the environment.
            total_positions = env.max_w * env.max_h
            stock_idx = action_index // (agent.max_product_type * total_positions)
            remainder = action_index % (agent.max_product_type * total_positions)
            product = remainder // total_positions
            pos_index = remainder % total_positions
            x = pos_index // env.max_h
            y = pos_index % env.max_h
            
            action = {
                "stock_idx": stock_idx,
                "product_idx": product,
                "size": np.array([0, 0]),  # The environment will determine the actual size.
                "position": np.array([x, y])
            }
            
            next_state, reward, done, truncated, _ = env.step(action)
            state_tensor = state_to_tensor(next_state, device)
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode+1}: Total Reward = {episode_reward}")
    
    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"Average Reward over {num_episodes} episodes: {avg_reward}")
    
    env.close()

if __name__ == "__main__":
    # Đường dẫn đến checkpoint bạn muốn đánh giá.
    checkpoint_file = "D:/A.I/Kỳ 8/REL301m_Final/cutting_stock_project/checkpoints/Case_optimized/dqn_checkpoint_ep1000_optimized.pt"  # hoặc checkpoint theo episode như "checkpoints/Case_new/dqn_checkpoint_ep50.pt"
    evaluate_agent(checkpoint_file, num_episodes=10)

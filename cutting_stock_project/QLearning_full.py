import os
import random
import pickle
import time
import numpy as np
import torch
import gymnasium as gym

# Import môi trường đã được fix (đảm bảo file này có trong thư mục env)
from env.CuttingStockEnvOptimized import CuttingStockEnvOptimized

############################################
# Q-Learning Agent với Q-table dạng dictionary
############################################

class QLearningAgent:
    def __init__(self, num_stocks, num_products, max_w, max_h,
                 learning_rate=0.1, discount_factor=0.95,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 device=None):
        self.num_stocks = num_stocks
        self.num_products = num_products
        self.max_w = max_w
        self.max_h = max_h
        # Số vị trí khả dĩ trong grid: max_w * max_h
        self.total_positions = max_w * max_h
        # Tổng số hành động: mỗi hành động được mã hóa bằng (stock_idx, product_idx, position)
        self.total_actions = num_stocks * num_products * self.total_positions
        
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Sử dụng GPU nếu có, nhưng khi lưu checkpoint sẽ chuyển về CPU
        self.device = torch.device('cuda' if (device is None and torch.cuda.is_available()) else 'cpu')
        
        # Q-table dưới dạng dictionary: key là state (dạng string hoặc tuple), value là tensor (cỡ: total_actions)
        self.q_table = {}
        
    def get_q_values(self, state_key):
        """Lấy Q-values của state_key, nếu chưa có khởi tạo với zeros."""
        if state_key not in self.q_table:
            self.q_table[state_key] = torch.zeros(self.total_actions, device=self.device)
        return self.q_table[state_key]
    
    def choose_action(self, state_key):
        """Chọn action theo chiến lược epsilon-greedy."""
        q_values = self.get_q_values(state_key)
        if random.random() < self.epsilon:
            return random.randint(0, self.total_actions - 1)
        else:
            return int(torch.argmax(q_values).item())
    
    def update(self, state_key, action, reward, next_state_key, done):
        """Cập nhật Q-table theo công thức Q-learning."""
        q_values = self.get_q_values(state_key)
        next_q_values = self.get_q_values(next_state_key)
        best_next = torch.max(next_q_values).item()
        td_target = reward + self.gamma * best_next * (not done)
        td_error = td_target - q_values[action].item()
        q_values[action] += self.alpha * td_error
        
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save_checkpoint(self, filename):
        """
        Trước khi lưu, chuyển toàn bộ các tensor Q-value về CPU và sang dtype float16
        để giảm dung lượng file.
        """
        q_table_cpu = {}
        for key, tensor in self.q_table.items():
            tensor_cpu = tensor.cpu()
            tensor_half = tensor_cpu.half()  # chuyển sang float16
            q_table_cpu[key] = tensor_half
        with open(filename, "wb") as f:
            pickle.dump(q_table_cpu, f)
        print(f"Checkpoint saved to {filename}")
    
    def load_checkpoint(self, filename):
        """
        Nạp Q-table từ file checkpoint, chuyển về self.device với dtype float32.
        """
        with open(filename, "rb") as f:
            loaded = pickle.load(f)
        self.q_table = {}
        for key, tensor in loaded.items():
            tensor_32 = tensor.to(dtype=torch.float32, device=self.device)
            self.q_table[key] = tensor_32
        print(f"Checkpoint loaded from {filename}")

############################################
# Helper functions
############################################

def get_state_key(observation):
    """
    Chuyển đổi quan sát từ môi trường thành key cho Q-table.
    Ở đây, ta sử dụng vector số lượng sản phẩm (observation["products"]) làm key.
    Bạn có thể cải tiến thêm bằng cách kết hợp các đặc trưng khác.
    """
    return str(observation["products"])

def get_env_action(action_index, agent):
    """
    Chuyển đổi chỉ số action (từ Q-table) thành action thực cho môi trường.
    Cách mã hóa:
      - stock_idx = action_index // (num_products * total_positions)
      - product_idx = (action_index % (num_products * total_positions)) // total_positions
      - pos_index = action_index % total_positions, từ đó x = pos_index // max_h, y = pos_index % max_h.
    """
    total_positions = agent.total_positions
    num_products = agent.num_products
    max_h = agent.max_h
    
    stock_idx = action_index // (num_products * total_positions)
    remainder = action_index % (num_products * total_positions)
    product_idx = remainder // total_positions
    pos_index = remainder % total_positions
    x = pos_index // max_h
    y = pos_index % max_h
    return {"stock_idx": stock_idx, "product_idx": product_idx, "position": (x, y)}

############################################
# Training Loop
############################################

def train_q_learning(num_episodes=500, checkpoint_interval=50, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Thiết lập tham số môi trường (có thể được lấy từ dữ liệu thực)
    grid_w = 50
    grid_h = 50
    num_stocks = 10
    num_products = 25
    
    # Khởi tạo môi trường với grid kích thước cố định (bạn có thể thay đổi dựa trên dữ liệu)
    env = CuttingStockEnvOptimized(render_mode="rgb_array",
                                   min_w=grid_w, min_h=grid_h,
                                   max_w=grid_w, max_h=grid_h,
                                   num_stocks=num_stocks,
                                   max_product_type=num_products,
                                   seed=42,
                                   max_steps=200)
    
    # Khởi tạo agent (sử dụng GPU nếu có; Q-table được lưu dạng dictionary)
    agent = QLearningAgent(num_stocks, num_products, grid_w, grid_h,
                           learning_rate=0.1, discount_factor=0.95,
                           epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01)
    
    # Training loop
    for episode in range(num_episodes):
        observation, info = env.reset()
        state_key = get_state_key(observation)
        ep_reward = 0
        done = False
        
        while not done:
            # Chọn hành động theo Q-table
            action_index = agent.choose_action(state_key)
            # Chuyển action_index thành action dictionary cho môi trường
            env_action = get_env_action(action_index, agent)
            next_observation, reward, done, _, info = env.step(env_action)
            next_state_key = get_state_key(next_observation)
            # Cập nhật Q-table
            agent.update(state_key, action_index, reward, next_state_key, done)
            state_key = next_state_key
            ep_reward += reward
        
        if (episode + 1) % checkpoint_interval == 0:
            filename = os.path.join(checkpoint_dir, f"q_table_checkpoint_ep{episode+1}.pkl")
            agent.save_checkpoint(filename)
            print(f"Episode {episode+1}: Total Reward = {ep_reward}, Epsilon = {agent.epsilon:.4f}")
        else:
            print(f"Episode {episode+1}: Total Reward = {ep_reward}, Epsilon = {agent.epsilon:.4f}")
    
    env.close()

if __name__ == "__main__":
    # Ví dụ: train 100 episodes, checkpoint mỗi 10 episodes
    train_q_learning(num_episodes=100, checkpoint_interval=10)

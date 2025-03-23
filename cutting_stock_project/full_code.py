import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import từ dự án
from env.CuttingStockEnvOptimized import CuttingStockEnvOptimized
from agents.q_learning_agent import QLearningAgent
from data.static_data import STATIC_DATA

def get_reward(observation, info):
    """
    Tính reward dựa trên filled_ratio, trim_loss và bonus cho số stock chưa dùng.
    """
    filled_ratio = info["filled_ratio"]
    trim_loss = info["trim_loss"]
    total_stocks = len(observation["stocks"])
    num_stocks_used = sum(1 for stock in observation["stocks"] if np.any(stock != -2))
    num_stocks_unused = total_stocks - num_stocks_used

    lambda_bonus = 0.2  # Hệ số bonus cho stock chưa dùng
    stock_bonus = lambda_bonus * (num_stocks_unused / total_stocks)

    return (filled_ratio - trim_loss) + stock_bonus

def compute_metrics(env, episode_steps, episode_reward):
    """
    Tính toán các metrics từ trạng thái cuối của môi trường.
    
    Trên mỗi stock:
      - valid_area: số ô hợp lệ (cell != -2)
      - free_area: số ô trống (cell == -1)
      - used_area: valid_area - free_area (nếu stock được dùng)
      - trim_loss: free_area / valid_area (nếu stock được dùng)
    
    Trả về dict chứa:
      - steps: số bước (episode_steps)
      - total_trim_loss: tổng trim loss của các stock đã dùng
      - remaining_stock: tổng số ô trống (free_area) của tất cả stock
      - used_stocks: số stock đã dùng (có cắt)
      - avg_used_stock_area: diện tích trung bình đã dùng trên các stock dùng
      - total_reward: reward của episode
    """
    used_stocks = int(np.sum(env.cutted_stocks))
    remaining_stock = 0
    used_areas = []
    total_trim_loss_val = 0

    for idx, stock in enumerate(env._stocks):
        valid_area = np.sum(stock != -2)
        free_area = np.sum(stock == -1)
        remaining_stock += free_area

        # Nếu stock này có cắt
        if env.cutted_stocks[idx] == 1:
            used_area = valid_area - free_area
            used_areas.append(used_area)
            trim_loss = free_area / valid_area if valid_area > 0 else 0
            total_trim_loss_val += trim_loss

    avg_used_stock_area = np.mean(used_areas) if used_areas else 0

    metrics = {
        "steps": episode_steps,
        "total_trim_loss": total_trim_loss_val,
        "remaining_stock": remaining_stock,
        "used_stocks": used_stocks,
        "avg_used_stock_area": avg_used_stock_area,
        "total_reward": episode_reward
    }
    return metrics

def train(num_episodes=10, state_size=100000, action_size=1000, max_w=50, max_h=50):
    """
    Train Q-Learning cho 10 batch có trong static_data.
    Mỗi batch:
      - Khởi tạo môi trường với max_w, max_h (fix).
      - Huấn luyện num_episodes.
      - Lưu checkpoint (Q-table) có reward cao nhất.
    Sau khi train xong, vẽ biểu đồ metrics và lưu file CSV với các cột:
      batch_id, steps, runtime, total_trim_loss, remaining_stock, used_stocks, avg_used_stock_area, total_reward
    """
    results = []

    # Đảm bảo thư mục lưu checkpoint tồn tại
    os.makedirs("checkpoints/q_learning/csv_train", exist_ok=True)

    # Lặp qua 10 batch (1 đến 10)
    for batch_id in range(1, 11):
        print(f"\n--- Training for Batch {batch_id} ---")
        static_config = STATIC_DATA[batch_id]

        # Khởi tạo môi trường
        env = CuttingStockEnvOptimized(
            render_mode=None,  # không hiển thị pygame window
            max_w=max_w,
            max_h=max_h,
            stock_list=static_config["stocks"],
            product_list=static_config["products"],
            seed=42
        )

        # Khởi tạo agent
        agent = QLearningAgent(
            state_size=state_size,
            action_size=action_size,
            alpha=0.1,
            gamma=0.9,
            epsilon=1.0,
            epsilon_decay=0.995,
            min_epsilon=0.01
        )

        best_reward = -np.inf
        best_metrics = None
        best_Q_table = None
        total_steps = 0

        batch_start_time = time.time()

        for episode in range(num_episodes):
            obs, info = env.reset(seed=42)
            state = agent.get_state(obs)
            ep_reward = 0
            episode_steps = 0
            done = False

            while not done:
                action_idx = agent.get_action(state)
                env_action = agent.get_env_action(action_idx, obs)
                obs, reward_terminal, terminated, truncated, info = env.step(env_action)

                # Tính reward theo hàm get_reward
                step_reward = get_reward(obs, info)
                ep_reward += step_reward

                next_state = agent.get_state(obs)
                agent.update(state, action_idx, step_reward, next_state)
                state = next_state

                episode_steps += 1
                total_steps += 1

                if terminated or truncated:
                    done = True

            print(f"Batch {batch_id} - Episode {episode}: Reward = {ep_reward:.4f}, Epsilon = {agent.epsilon:.4f}")

            # Nếu episode này có reward cao nhất, lưu checkpoint và metrics
            if ep_reward > best_reward:
                best_reward = ep_reward
                # Tính metrics
                current_metrics = compute_metrics(env, episode_steps, ep_reward)
                best_metrics = current_metrics
                best_Q_table = agent.Q_table.copy()

        batch_runtime = time.time() - batch_start_time
        if best_metrics is not None:
            best_metrics["runtime"] = batch_runtime
            best_metrics["batch_id"] = batch_id

        results.append(best_metrics)

        # Lưu checkpoint Q-table của batch (reward cao nhất)
        checkpoint_filename = f"checkpoints/q_learning/csv_train/q_table_checkpoint_batch{batch_id}.pkl"
        with open(checkpoint_filename, "wb") as f:
            pickle.dump(best_Q_table, f)
        print(f"Checkpoint for batch {batch_id} saved to {checkpoint_filename}")

        env.close()

    # Sau khi train xong tất cả batch, lưu metrics vào file CSV
    df = pd.DataFrame(results)
    # Sắp xếp cột theo yêu cầu
    df = df[["batch_id", "steps", "runtime", "total_trim_loss",
             "remaining_stock", "used_stocks", "avg_used_stock_area", "total_reward"]]
    csv_filename = "metrics.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\nMetrics saved to {csv_filename}")

    # Vẽ biểu đồ metrics
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    axs[0, 0].plot(df["batch_id"], df["total_reward"], marker='o')
    axs[0, 0].set_title("Total Reward vs Batch ID")
    axs[0, 0].set_xlabel("Batch ID")
    axs[0, 0].set_ylabel("Total Reward")

    axs[0, 1].plot(df["batch_id"], df["runtime"], marker='o')
    axs[0, 1].set_title("Runtime vs Batch ID")
    axs[0, 1].set_xlabel("Batch ID")
    axs[0, 1].set_ylabel("Runtime (s)")

    axs[1, 0].plot(df["batch_id"], df["steps"], marker='o')
    axs[1, 0].set_title("Steps vs Batch ID")
    axs[1, 0].set_xlabel("Batch ID")
    axs[1, 0].set_ylabel("Steps")

    axs[1, 1].plot(df["batch_id"], df["total_trim_loss"], marker='o')
    axs[1, 1].set_title("Total Trim Loss vs Batch ID")
    axs[1, 1].set_xlabel("Batch ID")
    axs[1, 1].set_ylabel("Total Trim Loss")

    plt.tight_layout()
    plt.savefig("metrics.png")
    plt.show()

if __name__ == "__main__":
    train()

import pickle
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Import lại môi trường và agent
from env.CuttingStockEnvOptimized import CuttingStockEnvOptimized
from agents.q_learning_agent import QLearningAgent

def load_q_table(agent, checkpoint_path):
    with open(checkpoint_path, "rb") as f:
        agent.q_table = pickle.load(f)
    print(f"Loaded Q-table from {checkpoint_path}")

def test_model(checkpoint_path, csv_file, render_mode="human"):
    """
    File test sau khi train:
      - Đọc file CSV chứa các batch (mỗi batch có các dòng dữ liệu về stock và product).
      - Với mỗi batch, tính kích thước grid dựa trên kích thước tối đa của stock trong batch.
      - Khởi tạo môi trường và ghi đè stocks/products theo dữ liệu.
      - Tạo agent, load Q-table từ checkpoint đã lưu.
      - Chạy thử (với render_mode có thể là "human" để hiển thị trực quan).
      - Thu thập các chỉ số (total_reward, runtime).
    """
    data = pd.read_csv(csv_file)
    batch_groups = data.groupby("batch_id")
    results = []
    
    for batch_id, group in batch_groups:
        # Lấy dữ liệu stock và product từ batch hiện tại
        stocks_df = group[group["type"] == "stock"]
        products_df = group[group["type"] == "product"]
        
        # Danh sách stocks: mỗi stock là tuple (width, height)
        stocks_list = [(int(r["width"]), int(r["height"])) for _, r in stocks_df.iterrows()]
        # Danh sách products: mỗi product là dict với kích thước và quantity=1
        products_list = [{"size": (int(r["width"]), int(r["height"])), "quantity": 1} 
                         for _, r in products_df.iterrows()]
        
        # Tính kích thước grid: dựa trên kích thước tối đa của stock trong batch
        grid_w = max(w for (w, _) in stocks_list) if stocks_list else 50
        grid_h = max(h for (_, h) in stocks_list) if stocks_list else 50
        num_stocks = len(stocks_list)
        num_products = len(products_list)
        
        print(f"Batch {batch_id}: grid size = ({grid_w}, {grid_h}), num_stocks = {num_stocks}, num_products = {num_products}")
        
        # Khởi tạo môi trường với tham số động dựa trên dữ liệu
        env = CuttingStockEnvOptimized(
            render_mode=render_mode,
            min_w=grid_w,
            min_h=grid_h,
            max_w=grid_w,
            max_h=grid_h,
            num_stocks=num_stocks,
            max_product_type=num_products,
            seed=42,
            max_steps=200
        )
        obs, info = env.reset()
        # Ghi đè stocks theo dữ liệu: mỗi stock là grid có kích thước (grid_w, grid_h)
        custom_stocks = []
        for (w, h) in stocks_list:
            stock = -2 * np.ones((grid_w, grid_h), dtype=int)
            stock[:w, :h] = -1
            custom_stocks.append(stock)
        env._stocks = tuple(custom_stocks)
        # Ghi đè products
        env._products = tuple(products_list)
        
        # Tạo agent, ép dùng CPU để tránh lỗi bộ nhớ GPU nếu cần
        agent = QLearningAgent(num_stocks, num_products, grid_w, grid_h,
                               learning_rate=0.1, discount_factor=0.95,
                               epsilon=0.0, epsilon_decay=0.995, epsilon_min=0.01)
        agent.device = torch.device("cpu")
        load_q_table(agent, checkpoint_path)
        
        # Chạy evaluation: thực hiện các step cho đến khi episode kết thúc (terminated)
        done = False
        total_reward = 0
        start_time = time.time()
        while not done:
            action = agent.choose_action(obs)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            # Nếu render_mode="human", cửa sổ pygame sẽ hiển thị trạng thái cắt
            time.sleep(0.05)
        runtime = time.time() - start_time
        
        print(f"Batch {batch_id}: Total Reward = {total_reward:.4f}, Runtime = {runtime:.2f}s")
        results.append({
            "batch_id": batch_id,
            "total_reward": total_reward,
            "runtime": runtime
        })
        
        env.close()
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Đường dẫn đến file checkpoint Q-table đã được train (ví dụ: checkpoint của episode 50)
    checkpoint_path = "checkpoints/q_learning/Case_test/q_table_checkpoint_ep80.pkl"
    # File CSV chứa tập dữ liệu test (ví dụ: data_custom.csv)
    csv_file = "data_custom.csv"
    
    # Chạy file test, với render_mode="human" để hiển thị trạng thái trực quan
    df_results = test_model(checkpoint_path, csv_file, render_mode="human")
    print(df_results)
    
    # Lưu kết quả test (nếu cần)
    df_results.to_csv("test_results.csv", index=False)

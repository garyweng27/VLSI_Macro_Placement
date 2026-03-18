import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# 引入我們設計的 Generator
from model_gnn import Generator

# ================= 設定參數 =================
NPZ_PATH = './processed_mgc_superblue14.npz'
MODEL_PATH = 'cgan_gnn_generator.pth'
Z_DIM = 64
# ==========================================

def main():
    # 1. 環境設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ 使用推論設備: {device}")

    # 2. 載入資料 (只需要拿第一筆資料的尺寸與連接矩陣來當條件)
    print(f"📂 讀取資料集 {NPZ_PATH}...")
    raw_data = np.load(NPZ_PATH, allow_pickle=True)
    dataset = raw_data['data']
    adj_matrix_np = raw_data['connectivity']
    
    sample = dataset[0]
    sizes_np = sample['feat']
    num_macros = sizes_np.shape[0]

    # 轉成 Tensor
    sizes = torch.tensor(sizes_np, dtype=torch.float32).unsqueeze(0).to(device)
    adj = torch.tensor(adj_matrix_np, dtype=torch.float32).unsqueeze(0).to(device)

    # 3. 載入模型 (僅做 Inference)
    print("🧠 載入 Generator 模型權重...")
    G = Generator(num_macros, z_dim=Z_DIM).to(device)
    # 讀取剛剛訓練好的 pth 檔案
    G.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    # 切換到推論模式
    G.eval()

    # 4. 產生佈局
    print("🎨 開始生成 Layout...")
    z = torch.randn(1, num_macros, Z_DIM, device=device)
    
    # 推論時不需要計算梯度
    with torch.no_grad():
        gen_pos = G(z, sizes, adj)
    
    # 把結果搬回 CPU 並轉成 numpy 陣列以便畫圖
    pos_np = gen_pos.squeeze(0).cpu().numpy()
    # ==========================================
    # 🛡️ 邊界合法化 (Boundary Legalization)
    # 將所有元件強制拉回畫布的安全範圍內，避免出界
    # ==========================================
    half_w = sizes_np[:, 0] / 2.0
    half_h = sizes_np[:, 1] / 2.0
    
    # 使用 np.clip 限制 x 和 y 座標的上下限
    pos_np[:, 0] = np.clip(pos_np[:, 0], half_w, 1.0 - half_w)
    pos_np[:, 1] = np.clip(pos_np[:, 1], half_h, 1.0 - half_h)

    # 5. 視覺化繪圖
    print("📈 繪製圖形中...")
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # 先畫出連線 (掃描 adjacency matrix 的 row 和 column)
    for row in range(num_macros):
        for column in range(num_macros):
            # 只要 row 和 column 之間有大於 0 的連線權重，且為了避免重複畫，只畫上半三角
            if adj_matrix_np[row, column] > 0 and row < column:
                x_values = [pos_np[row, 0], pos_np[column, 0]]
                y_values = [pos_np[row, 1], pos_np[column, 1]]
                ax.plot(x_values, y_values, color='gray', linestyle='-', linewidth=0.5, alpha=0.4)

    # 再畫出 Macro 的矩形
    for i in range(num_macros):
        w, h = sizes_np[i]
        cx, cy = pos_np[i]
        
        # matplotlib 畫矩形需要左下角座標，所以要把中心點轉換一下
        bl_x = cx - w / 2
        bl_y = cy - h / 2
        
        rect = patches.Rectangle((bl_x, bl_y), w, h, linewidth=1.5, edgecolor='darkblue', facecolor='skyblue', alpha=0.7)
        ax.add_patch(rect)
        
        # 在中心標上 Macro 的編號
        ax.text(cx, cy, str(i), color='black', fontsize=9, ha='center', va='center', fontweight='bold')

    # 設定圖表外觀
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title('GNN Generated Layout (mgc_superblue12 - 46 Macros)', fontsize=16)
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 存檔
    save_filename = 'generated_superblue12.png'
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    print(f"🎉 圖片已成功儲存為 {save_filename}！")

if __name__ == "__main__":
    main()
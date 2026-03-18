import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from model import Generator

# ================= 設定參數 =================
NPZ_PATH = './processed_mgc_pci_bridge32_a.npz'
MODEL_PATH = 'cgan_generator.pth'
Z_DIM = 64
CANDIDATE_COUNT = 100  # 一次生成 100 張，從中選最好的
# ==========================================

def calc_hpwl(pos, adj_matrix):
    """計算線長"""
    loss = 0
    num_macros = pos.shape[0]
    for i in range(num_macros):
        for j in range(i + 1, num_macros):
            weight = adj_matrix[i, j]
            if weight > 0:
                dist = np.abs(pos[i] - pos[j]).sum()
                loss += dist * weight
    return loss

def calc_overlap_score(pos, sizes):
    """
    極速版重疊計算 (Vectorized)
    完全不使用 Python For-Loop，速度快 100 倍以上
    """
    # pos, sizes: [N, 2]
    N = pos.shape[0]
    
    # 利用廣播 (Broadcasting) 擴展維度
    # xi: [N, 1], xj: [1, N]
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    w = sizes[:, 0:1]
    h = sizes[:, 1:2]
    
    # 一次計算所有 pair 的距離矩陣 [N, N]
    dx = np.abs(x - x.T)
    dy = np.abs(y - y.T)
    
    # 一次計算所有 pair 的最小安全距離
    min_dist_x = (w + w.T) / 2
    min_dist_y = (h + h.T) / 2
    
    # 計算重疊量 (ReLU)
    overlap_w = np.maximum(0, min_dist_x - dx)
    overlap_h = np.maximum(0, min_dist_y - dy)
    
    area_matrix = overlap_w * overlap_h
    
    # 只取上三角矩陣 (Upper Triangle)，避免重複算和算到自己
    # k=1 代表不包含對角線 (自己跟自己重疊不算)
    mask = np.triu(np.ones((N, N)), k=1)
    
    total_overlap = (area_matrix * mask).sum()
    
    return total_overlap
def legalize_positions(pos, sizes):
    """
    合法化：將跑出邊界的 Macro 強制推回畫布內
    pos: [N, 2] (cx, cy)
    sizes: [N, 2] (w, h)
    """
    # 複製一份，避免改到原始資料
    legal_pos = np.copy(pos)
    
    # 畫布範圍是 0 ~ 1
    # 限制 x 的範圍：左邊不能小於 w/2，右邊不能大於 1-w/2
    min_x = sizes[:, 0] / 2
    max_x = 1.0 - (sizes[:, 0] / 2)
    legal_pos[:, 0] = np.clip(legal_pos[:, 0], min_x, max_x)
    
    # 限制 y 的範圍：下邊不能小於 h/2，上邊不能大於 1-h/2
    min_y = sizes[:, 1] / 2
    max_y = 1.0 - (sizes[:, 1] / 2)
    legal_pos[:, 1] = np.clip(legal_pos[:, 1], min_y, max_y)
    
    return legal_pos
def draw_layout_with_connections(ax, positions, sizes, adj_matrix, title, color='blue'):
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    
    # 畫連線
    num_macros = len(positions)
    max_weight = adj_matrix.max() if adj_matrix.max() > 0 else 1
    for i in range(num_macros):
        for j in range(i + 1, num_macros):
            weight = adj_matrix[i, j]
            if weight > 0:
                p1, p2 = positions[i], positions[j]
                lw = 1 + (weight / max_weight) * 3 
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='gray', alpha=0.5, linewidth=lw, linestyle='--')

    # 畫 Macro
    for i in range(num_macros):
        cx, cy = positions[i]
        w, h = sizes[i]
        rect = patches.Rectangle((cx - w/2, cy - h/2), w, h, linewidth=2, edgecolor='black', facecolor=color, alpha=0.8)
        ax.add_patch(rect)
        ax.text(cx, cy, str(i), color='white', ha='center', va='center', fontweight='bold')
    ax.add_patch(patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', linewidth=2))

def main():
    # 1. 載入數據
    print(f"📂 Loading data from {NPZ_PATH}...")
    raw_data = np.load(NPZ_PATH, allow_pickle=True)
    dataset = raw_data['data']
    adj_matrix = raw_data['connectivity']
    sample_pos = dataset[0]['pos']
    print(f"😱 檢查 Macro 數量: {sample_pos.shape[0]}")
    
    sample = dataset[0]
    real_pos = sample['pos']
    cond_sizes = sample['feat'] # [N, 2]
    
    # 2. 載入模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available(): device = torch.device("mps")

    num_macros = real_pos.shape[0]
    G = Generator(num_macros, z_dim=Z_DIM).to(device)
    G.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    G.eval()

    # 3. [關鍵修改] 批量生成並挑選最佳結果
    print(f"🎰 Generating {CANDIDATE_COUNT} candidates to find the best layout...")
    
    # 一次生成 100 個 Batch
    z = torch.randn(CANDIDATE_COUNT, num_macros, Z_DIM).to(device)
    
    # 條件要複製 100 份
    cond_tensor = torch.tensor(cond_sizes, dtype=torch.float32).to(device) # [N, 2]
    cond_batch = cond_tensor.unsqueeze(0).repeat(CANDIDATE_COUNT, 1, 1) # [100, N, 2]
    
    with torch.no_grad():
        fake_pos_batch = G(z, cond_batch).cpu().numpy() # [100, N, 2]

    # 4. 評分機制：挑選 Overlap 最小，且 HPWL 合理的
    best_idx = -1
    min_overlap = float('inf')
    best_hpwl = float('inf')

    for i in range(CANDIDATE_COUNT):
        current_pos = fake_pos_batch[i]
        
        # 算出重疊分數
        overlap = calc_overlap_score(current_pos, cond_sizes)
        hpwl = calc_hpwl(current_pos, adj_matrix)
        
        # 挑選邏輯：優先選重疊最小的；如果重疊一樣小，選線長最短的
        if overlap < min_overlap:
            min_overlap = overlap
            best_idx = i
            best_hpwl = hpwl
        elif abs(overlap - min_overlap) < 1e-6: # 如果重疊差不多 (例如都是 0)
            if hpwl < best_hpwl:
                best_hpwl = hpwl
                best_idx = i

    print(f"🏆 Best Candidate Found (Index {best_idx}):")
    print(f"   Overlap Score: {min_overlap:.6f} (Should be close to 0)")
    print(f"   HPWL Score:    {best_hpwl:.4f}")

    raw_fake_pos = fake_pos_batch[best_idx]
    
    # [新增] 執行合法化，把跑出去的推回來
    final_fake_pos = legalize_positions(raw_fake_pos, cond_sizes)
    
    # [選用] 因為推回來後可能會稍微改變分數，可以重新算一下顯示給你看
    final_hpwl = calc_hpwl(final_fake_pos, adj_matrix)
    print(f"   (Legalized HPWL: {final_hpwl:.4f})")
    
    # 5. 畫圖
    score_real = calc_hpwl(real_pos, adj_matrix)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    draw_layout_with_connections(axes[0], real_pos, cond_sizes, adj_matrix, 
                                 f"Real Layout\nHPWL: {score_real:.1f}", color='green')
    draw_layout_with_connections(axes[1], final_fake_pos, cond_sizes, adj_matrix, 
                                 f"AI Generated (Best of {CANDIDATE_COUNT})\nHPWL: {best_hpwl:.1f}\nOverlap: {min_overlap:.4f}", color='red')

    plt.tight_layout()
    plt.savefig("result_best_of_N.png")
    print("\n✅ Result saved to 'result_best_of_N.png'")
    plt.show()

if __name__ == "__main__":
    main()
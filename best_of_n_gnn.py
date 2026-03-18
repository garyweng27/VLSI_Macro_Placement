import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from model_gnn import Generator

# ================= 設定參數 =================
NPZ_PATH = './processed_mgc_superblue14.npz'
MODEL_PATH = 'cgan_gnn_generator.pth'
Z_DIM = 64
NUM_CANDIDATES = 10000  # 🚀 RTX 3090 專屬：平行產生 1 萬種排版！
# ==========================================

def evaluate_candidates(pos, sizes, adj):
    """
    全向量化評分函數：一次計算 10,000 個候選者的 Wirelength 和 Overlap
    pos shape: [10000, 46, 2]
    """
    # 1. 計算 Wirelength (HPWL)
    dx = torch.abs(pos[:, :, 0].unsqueeze(2) - pos[:, :, 0].unsqueeze(1))
    dy = torch.abs(pos[:, :, 1].unsqueeze(2) - pos[:, :, 1].unsqueeze(1))
    dist = dx + dy
    # 沿著 row 和 column 維度加總，保留 batch 維度
    wire_loss = torch.sum(adj * dist, dim=(1, 2)) / 2.0

    # 2. 計算 Overlap Area
    x = pos[:, :, 0].unsqueeze(2)
    y = pos[:, :, 1].unsqueeze(2)
    w = sizes[:, :, 0].unsqueeze(2)
    h = sizes[:, :, 1].unsqueeze(2)
    
    dx_over = torch.abs(x - x.transpose(1, 2))
    dy_over = torch.abs(y - y.transpose(1, 2))
    
    min_dist_x = (w + w.transpose(1, 2)) / 2.0
    min_dist_y = (h + h.transpose(1, 2)) / 2.0
    
    overlap_w = torch.relu(min_dist_x - dx_over)
    overlap_h = torch.relu(min_dist_y - dy_over)
    
    area = overlap_w * overlap_h
    mask = 1.0 - torch.eye(pos.size(1), device=pos.device).unsqueeze(0)
    
    overlap_loss = torch.sum(area * mask, dim=(1, 2)) / 2.0
    
    return wire_loss, overlap_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ 啟動 Best-of-N 推論引擎，使用設備: {device}")

    raw_data = np.load(NPZ_PATH, allow_pickle=True)
    sizes_np = raw_data['data'][0]['feat']
    adj_matrix_np = raw_data['connectivity']
    num_macros = sizes_np.shape[0]

    sizes = torch.tensor(sizes_np, dtype=torch.float32, device=device)
    adj = torch.tensor(adj_matrix_np, dtype=torch.float32, device=device)

    G = Generator(num_macros, z_dim=Z_DIM).to(device)
    G.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    G.eval()

    print(f"🌀 正在平行生成 {NUM_CANDIDATES} 個候選佈局...")
    with torch.no_grad():
        # 一次產生 10,000 個雜訊
        z = torch.randn(NUM_CANDIDATES, num_macros, Z_DIM, device=device)
        
        sizes_batch = sizes.unsqueeze(0).expand(NUM_CANDIDATES, -1, -1)
        adj_batch = adj.unsqueeze(0).expand(NUM_CANDIDATES, -1, -1)
        
        # 讓 GPU 瞬間算完 1 萬次 GNN 前向傳播
        gen_pos = G(z, sizes_batch, adj_batch)

        # ==========================================
        # 🛡️ 嚴格的邊界合法化 (Boundary Legalization)
        # 必須在評分「前」執行，確保所有候選者都在畫布內
        # ==========================================
        half_w = sizes_batch[:, :, 0] / 2.0
        half_h = sizes_batch[:, :, 1] / 2.0
        gen_pos[:, :, 0] = torch.max(half_w, torch.min(gen_pos[:, :, 0], 1.0 - half_w))
        gen_pos[:, :, 1] = torch.max(half_h, torch.min(gen_pos[:, :, 1], 1.0 - half_h))

        print("⚖️ 正在對所有候選者進行嚴格評分 (Overlap & HPWL)...")
        wire_scores, overlap_scores = evaluate_candidates(gen_pos, sizes_batch, adj_batch)
        
        # 設計綜合分數：重疊是死罪 (權重極高)，線長是加分項
        # Score = Overlap * 10000.0 + HPWL
        total_scores = (overlap_scores * 10000.0) + wire_scores
        
        # 找出綜合分數最低 (最好) 的那一個候選者
        best_idx = torch.argmin(total_scores).item()
        best_pos = gen_pos[best_idx].cpu().numpy()
        
        print("\n🏆 最佳候選者脫穎而出！")
        print(f"   - 候選者編號: #{best_idx}")
        print(f"   - 重疊面積: {overlap_scores[best_idx].item():.6f}")
        print(f"   - 總線長 (HPWL): {wire_scores[best_idx].item():.4f}")

    # 繪製最佳結果
    print("📈 繪製圖形中...")
    fig, ax = plt.subplots(figsize=(12, 12))
    
    for r in range(num_macros):
        for c in range(num_macros):
            if adj_matrix_np[r, c] > 0 and r < c:
                x_values = [best_pos[r, 0], best_pos[c, 0]]
                y_values = [best_pos[r, 1], best_pos[c, 1]]
                ax.plot(x_values, y_values, color='gray', linestyle='-', linewidth=0.5, alpha=0.4)

    for i in range(num_macros):
        w, h = sizes_np[i]
        cx, cy = best_pos[i]
        bl_x = cx - w / 2
        bl_y = cy - h / 2
        rect = patches.Rectangle((bl_x, bl_y), w, h, linewidth=1.5, edgecolor='darkblue', facecolor='skyblue', alpha=0.7)
        ax.add_patch(rect)
        ax.text(cx, cy, str(i), color='black', fontsize=9, ha='center', va='center', fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title(f'GNN Layout (Best of {NUM_CANDIDATES} Candidates)', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    save_filename = 'best_gnn_layout.png'
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    print(f"🎉 最優圖片已成功儲存為 {save_filename}！")

if __name__ == "__main__":
    main()
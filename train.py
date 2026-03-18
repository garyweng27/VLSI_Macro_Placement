import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MacroPlacementDataset
from model import Generator, Discriminator
import numpy as np

# ================= 設定參數 =================
NPZ_PATH = './processed_mgc_pci_bridge32_a.npz'
BATCH_SIZE = 8  # 因為數據少，Batch Size 不要設太大
LR = 0.0002     # Learning Rate
Z_DIM = 64      # 雜訊維度
EPOCHS = 2000   # 訓練次數 (因為數據少，可以跑多一點)
# ==========================================
def calc_wirelength_loss(pos, adj_matrix):
    # pos: [Batch, N, 2]
    # adj_matrix: [N, N] (Pre-calculated connectivity)
    
    loss = 0
    num_macros = pos.size(1)
    
    # 遍歷所有的 Macro 對 (i, j)
    # 為了效率，這可以用矩陣操作優化，但我們先寫 Loop 比較好懂
    for i in range(num_macros):
        for j in range(i + 1, num_macros):
            weight = adj_matrix[i, j]
            if weight > 0:
                # 計算距離 (曼哈頓距離 Manhattan Distance) |x1-x2| + |y1-y2|
                # 這是 VLSI 佈局中最常用的距離算法
                dist = torch.abs(pos[:, i, :] - pos[:, j, :]).sum(dim=1) # [Batch]
                
                # Loss = 距離 * 權重 (連線數)
                loss += (dist * weight).mean()
                
    return loss
def calc_overlap_loss(pos, size):
    """
    計算重疊懲罰 (Overlap Penalty)
    pos: [Batch, N, 2] (中心點座標 cx, cy)
    size: [Batch, N, 2] (寬高 w, h)
    """
    loss = 0
    num_macros = pos.size(1)
    
    # 遍歷所有 Macro 對 (i, j)
    for i in range(num_macros):
        for j in range(i + 1, num_macros):
            
            # 1. 取得兩個 Macro 的資訊
            # pos 是中心點，size 是全寬全高
            xi, yi = pos[:, i, 0], pos[:, i, 1]
            wi, hi = size[:, i, 0], size[:, i, 1]
            
            xj, yj = pos[:, j, 0], pos[:, j, 1]
            wj, hj = size[:, j, 0], size[:, j, 1]
            
            # 2. 計算 X 軸和 Y 軸的距離
            dx = torch.abs(xi - xj)
            dy = torch.abs(yi - yj)
            
            # 3. 計算「允許的最小距離」(兩者半寬/半高之和)
            # 如果距離小於這個值，代表撞到了
            min_dist_x = (wi + wj) / 2
            min_dist_y = (hi + hj) / 2
            
            # 4. 計算重疊量 (ReLU 把負值變成 0，代表沒重疊)
            overlap_w = torch.relu(min_dist_x - dx)
            overlap_h = torch.relu(min_dist_y - dy)
            
            # 5. 重疊面積 = 寬重疊 * 高重疊
            area = (overlap_w * overlap_h) ** 2
            
            # 加總所有 Batch 的重疊面積
            loss += area.sum()
            
    return loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_data = np.load(NPZ_PATH, allow_pickle=True)
    adj_matrix_np = raw_data['connectivity']
    adj_matrix = torch.tensor(adj_matrix_np, dtype=torch.float32).to(device)
    # 1. 準備數據
    dataset = MacroPlacementDataset(NPZ_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 取得 Macro 數量 (從數據中自動偵測，這裡是 4)
    sample_pos, sample_feat = dataset[0]
    num_macros = sample_pos.shape[0] 
    print(f"🎯 Target: Placing {num_macros} Macros")

    # 2. 初始化模型

    # 如果你是 Mac M1/M2，可以用 "mps" 加速
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"⚙️ Using device: {device}")

    G = Generator(num_macros, z_dim=Z_DIM).to(device)
    D = Discriminator(num_macros).to(device)

    # 3. 優化器 (Adam 是 GAN 的標準配備)
    optimizer_G = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

    criterion = torch.nn.BCELoss() # Binary Cross Entropy

    # 4. 訓練迴路
    print("🚀 Start Training...")
    
    for epoch in range(EPOCHS):
        for i, (real_pos, condition) in enumerate(dataloader):
            
            batch_size = real_pos.size(0)
            real_pos = real_pos.to(device)
            condition = condition.to(device)
            
            # 建立標籤 (1=真, 0=假)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # ---------------------
            #  訓練 Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # (1) 讓 D 看真實數據
            outputs_real = D(real_pos, condition)
            loss_D_real = criterion(outputs_real, real_labels)
            
            # (2) 讓 D 看生成數據 (Fake)
            z = torch.randn(batch_size, num_macros, Z_DIM).to(device)
            fake_pos = G(z, condition)
            outputs_fake = D(fake_pos.detach(), condition) # detach 防止更新 G
            loss_D_fake = criterion(outputs_fake, fake_labels)
            
            loss_D = loss_D_real + loss_D_fake
            loss_D.backward()
            optimizer_D.step()

            # -----------------
            #  訓練 Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # 騙過 D (希望 D 覺得這是真的)
            outputs_fake = D(fake_pos, condition)
            loss_G_gan = criterion(outputs_fake, real_labels) # 1. 騙過 Discriminator
            
            loss_G_wire = calc_wirelength_loss(fake_pos, adj_matrix) # 2. 線長越短越好
            
            # [新增] 3. 重疊越少越好
            loss_G_overlap = calc_overlap_loss(fake_pos, condition) 

            # 總 Loss 配方 (權重調整是玄學，這是經驗值)
            # 建議：Overlap 的懲罰要很重 (x10)，因為重疊是絕對不允許的
            if epoch < 500:
                # 前 500 次訓練：只看重疊，不看線長 (Wire weight = 0)
                # 讓 AI 專心學會「把東西分開擺」
                w_wire = 0.0
                w_overlap = 100.0
            else:
                # 500 次之後：AI 已經學會不重疊了，開始慢慢加入線長要求
                w_wire = 0.05
                w_overlap = 100.0 # 保持高壓
                
            loss_G = loss_G_gan + (w_wire * loss_G_wire) + (w_overlap * loss_G_overlap)
            loss_G.backward()
            optimizer_G.step()

        # 每 100 epoch 印出一次進度
        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{EPOCHS}]")
            print(f"   Loss D: {loss_D.item():.4f}")
            print(f"   Loss G Total: {loss_G.item():.4f}")
            print(f"      - GAN: {loss_G_gan.item():.4f}")
            print(f"      - Wire: {loss_G_wire.item():.4f}")
            print(f"      - Overlap: {loss_G_overlap.item():.4f}")
            G.eval()
            # 簡單看一下 G 目前生成的第一個 macro 座標
            with torch.no_grad():
                test_z = torch.randn(1, num_macros, Z_DIM).to(device)
                test_cond = condition[0:1] # 拿第一筆資料的尺寸當測試
                generated = G(test_z, test_cond).cpu().numpy()[0]
                print(f"   Sample G Output (Macro 0 pos): {generated[0]}")
            G.train()
    # 5. 存檔
    torch.save(G.state_dict(), "cgan_generator.pth")
    print("💾 Model saved to cgan_generator.pth")

if __name__ == "__main__":
    main()
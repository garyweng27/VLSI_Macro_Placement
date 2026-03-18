import torch
from torch.utils.data import Dataset
import numpy as np

class MacroPlacementDataset(Dataset):
    def __init__(self, npz_path):
        # 1. 載入我們剛剛做好的數據
        print(f"📂 Loading dataset from {npz_path}...")
        raw_data = np.load(npz_path, allow_pickle=True)
        self.data_list = raw_data['data'] # List of dicts {'pos':..., 'feat':...}
        self.canvas_size = raw_data['canvas_size']
        
        # 2. 轉換成 Tensor 格式
        self.samples = []
        for item in self.data_list:
            # pos: [N, 2], feat: [N, 2]
            # 我們把它們拼起來方便處理 [N, 4] -> (x, y, w, h)
            pos = torch.tensor(item['pos'], dtype=torch.float32)
            feat = torch.tensor(item['feat'], dtype=torch.float32)
            self.samples.append((pos, feat))
            
        print(f"✅ Loaded {len(self.samples)} samples.")
        print(f"   Macros per sample: {self.samples[0][0].shape[0]}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 取得原始數據
        pos, feat = self.samples[idx]
        
        # =========================================
        # 🔧 Data Augmentation (數據增強)
        # 隨機翻轉，增加數據多樣性
        # =========================================
        
        # 1. 隨機水平翻轉 (Horizontal Flip)
        if torch.rand(1) < 0.5:
            # x 座標變成 1.0 - x
            # 注意：因為 x 是左下角座標，翻轉後要考慮寬度
            # 新 x = 1.0 - (原 x + w) => 1.0 - x - w
            # 簡化版(如果是中心點座標): x = 1.0 - x
            # 假設我們之前存的是「中心點」座標 (preprocess_data.py 裡寫的是 cx, cy)
            pos = pos.clone()
            pos[:, 0] = 1.0 - pos[:, 0]

        # 2. 隨機垂直翻轉 (Vertical Flip)
        if torch.rand(1) < 0.5:
            pos = pos.clone()
            pos[:, 1] = 1.0 - pos[:, 1]
            
        # 3. 隨機加入微小雜訊 (Jitter)
        # 模擬電路佈局的微小變動，防止 Overfit
        noise = torch.randn_like(pos) * 0.01
        pos = pos + noise
        
        # 確保座標還是在 0~1 之間 (Clamp)
        pos = torch.clamp(pos, 0.0, 1.0)

        # 回傳:
        # real_pos: 真實座標 (Ground Truth) [N, 2]
        # condition: Macro 的寬高特徵 [N, 2]
        return pos, feat
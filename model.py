import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, num_macros, z_dim=64, cond_dim=2):
        super(Generator, self).__init__()
        
        self.num_macros = num_macros
        self.z_dim = z_dim # 雜訊維度
        
        # 輸入維度 = (雜訊 + 條件(w,h)) * Macro數量
        input_dim = (z_dim + cond_dim) * num_macros
        
        # 簡單的 MLP 架構
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            
            nn.Linear(512, num_macros * 2), # 輸出每個 Macro 的 (x, y)
            nn.Sigmoid() # 限制輸出在 0~1 之間 (因為我們做過 Normalize)
        )

    def forward(self, z, condition):
        # z: [Batch, num_macros, z_dim]
        # condition: [Batch, num_macros, 2] (width, height)
        
        batch_size = z.size(0)
        
        # 把輸入攤平 (Flatten) 餵給 MLP
        # Input shape: [Batch, num_macros * (z_dim + 2)]
        x = torch.cat([z, condition], dim=2) 
        x = x.view(batch_size, -1) 
        
        out = self.net(x)
        
        # 重塑回 [Batch, num_macros, 2]
        return out.view(batch_size, self.num_macros, 2)

class Discriminator(nn.Module):
    def __init__(self, num_macros, cond_dim=2):
        super(Discriminator, self).__init__()
        
        # 輸入 = (座標(x,y) + 條件(w,h)) * Macro數量
        input_dim = (2 + cond_dim) * num_macros
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(256, 1), # 輸出一個分數 (Real or Fake)
            nn.Sigmoid()
        )

    def forward(self, pos, condition):
        # pos: [Batch, num_macros, 2]
        # condition: [Batch, num_macros, 2]
        
        batch_size = pos.size(0)
        
        x = torch.cat([pos, condition], dim=2)
        x = x.view(batch_size, -1)
        
        return self.net(x)
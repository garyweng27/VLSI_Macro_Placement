import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from model_gnn import Generator

def calc_wirelength_loss(pos, adj):
    dx = torch.abs(pos[:, :, 0].unsqueeze(2) - pos[:, :, 0].unsqueeze(1))
    dy = torch.abs(pos[:, :, 1].unsqueeze(2) - pos[:, :, 1].unsqueeze(1))
    dist = dx + dy
    degree = torch.sum(adj, dim=2, keepdim=True) + 1.0
    norm_adj = adj / torch.sqrt(torch.matmul(degree, degree.transpose(1, 2)))
    return torch.sum(norm_adj * dist) / 2.0 / pos.size(0)

def calc_local_overlap_loss(pos, sizes, device):
    N = pos.size(1)
    x = pos[:, :, 0].unsqueeze(2)
    y = pos[:, :, 1].unsqueeze(2)
    
    noise_x = torch.randn_like(x) * 1e-5
    noise_y = torch.randn_like(y) * 1e-5
    x_jitter = x + noise_x
    y_jitter = y + noise_y
    
    w = sizes[:, :, 0].unsqueeze(2)
    h = sizes[:, :, 1].unsqueeze(2)
    
    dx = torch.sqrt((x_jitter - x_jitter.transpose(1, 2))**2 + 1e-8)
    dy = torch.sqrt((y_jitter - y_jitter.transpose(1, 2))**2 + 1e-8)
    
    min_dist_x = (w + w.transpose(1, 2)) / 2.0
    min_dist_y = (h + h.transpose(1, 2)) / 2.0
    
    overlap_w = torch.relu(min_dist_x - dx)
    overlap_h = torch.relu(min_dist_y - dy)
    
    min_w = torch.min(w, w.transpose(1, 2))
    min_h = torch.min(h, h.transpose(1, 2))
    
    rel_overlap_w = overlap_w / (min_w + 1e-5)
    rel_overlap_h = overlap_h / (min_h + 1e-5)
    
    repulsion = rel_overlap_w * rel_overlap_h
    mask = 1.0 - torch.eye(N, device=device).unsqueeze(0)
    
    return torch.sum(repulsion * mask) / 2.0 / pos.size(0)

def calc_boundary_loss(pos, sizes):
    half_w = sizes[:, :, 0] / 2.0
    half_h = sizes[:, :, 1] / 2.0
    loss_x = torch.relu(half_w - pos[:, :, 0]) + torch.relu(pos[:, :, 0] - (1.0 - half_w))
    loss_y = torch.relu(half_h - pos[:, :, 1]) + torch.relu(pos[:, :, 1] - (1.0 - half_h))
    return torch.sum(loss_x + loss_y) / pos.size(0)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_data = np.load('./processed_mgc_superblue14.npz', allow_pickle=True)
    dataset = raw_data['data']
    adj_matrix = raw_data['connectivity']
    
    num_macros = dataset[0]['pos'].shape[0]
    
    sizes_tensor = torch.tensor(dataset[0]['feat'], dtype=torch.float32, device=device).unsqueeze(0)
    adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32, device=device).unsqueeze(0)
    
    G = Generator(num_macros, 64).to(device)
    
    optimizer_G = optim.AdamW(G.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=500, gamma=0.5)
    
    start_time = time.time()
    
    fixed_z = torch.zeros(1, num_macros, 64, device=device)
    
    for epoch in range(1, 3001):
        if epoch <= 1000:
            w_wire, w_overlap, w_bound = 0.05, 10.0, 50.0
        elif epoch <= 2500:
            w_wire, w_overlap, w_bound = 0.5, 10.0, 100.0
        else:
            w_wire, w_overlap, w_bound = 0.001, 50.0, 200.0
            
        optimizer_G.zero_grad()
        
        fake_pos = G(fixed_z, sizes_tensor, adj_tensor)
        
        g_loss_wire = calc_wirelength_loss(fake_pos, adj_tensor)
        g_loss_overlap = calc_local_overlap_loss(fake_pos, sizes_tensor, device)
        g_loss_bound = calc_boundary_loss(fake_pos, sizes_tensor)
        
        g_loss = (w_wire * g_loss_wire) + (w_overlap * g_loss_overlap) + (w_bound * g_loss_bound)
        g_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0)
        
        optimizer_G.step()
        scheduler_G.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Loss: {g_loss.item():.4f} | Wire: {g_loss_wire.item():.4f} | Overlap: {g_loss_overlap.item():.4f}")
            
    end_time = time.time()
    torch.save(G.state_dict(), 'cgan_gnn_generator.pth')
    print(f"Time: {end_time - start_time:.4f} seconds")

if __name__ == '__main__':
    main()
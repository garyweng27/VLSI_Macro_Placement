import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseGCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(DenseGCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x, adj):
        h = self.linear(x)
        out = torch.matmul(adj, h)
        return self.norm(F.relu(out))

class Generator(nn.Module):
    def __init__(self, num_macros, z_dim=64):
        super(Generator, self).__init__()
        self.num_macros = num_macros
        self.node_emb = nn.Embedding(num_macros, z_dim)
        nn.init.xavier_uniform_(self.node_emb.weight)
        
        in_dim = z_dim * 2 + 2
        self.gcn1 = DenseGCNLayer(in_dim, 128)
        self.gcn2 = DenseGCNLayer(128, 64)
        self.linear_out = nn.Linear(64 + z_dim, 2)

    def forward(self, z, sizes, adj):
        device = sizes.device
        identity = torch.eye(self.num_macros, device=device).unsqueeze(0)
        adj_hat = adj + identity
        
        degree = torch.sum(adj_hat, dim=2)
        d_inv_sqrt = torch.pow(degree, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat = torch.diag_embed(d_inv_sqrt)
        norm_adj = torch.matmul(torch.matmul(d_mat, adj_hat), d_mat)
        
        nodes = torch.arange(self.num_macros, device=device).unsqueeze(0).expand(sizes.size(0), -1)
        emb = self.node_emb(nodes)
        
        x = torch.cat([z, emb, sizes], dim=-1)
        x = self.gcn1(x, norm_adj)
        x = self.gcn2(x, norm_adj)
        
        x_final = torch.cat([x, emb], dim=-1)
        logits = self.linear_out(x_final)
        
        return (torch.sigmoid(logits) * 1.2) - 0.1
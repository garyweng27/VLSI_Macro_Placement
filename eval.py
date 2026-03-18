import torch
import numpy as np

import time
from model_gnn import Generator


def x_y_z_1(
    q,
      w_1
):

  a_1 = torch.abs(q[:, :, 0].unsqueeze(2) - q[:, :, 0].unsqueeze(1))
  b_2 = torch.abs(q[:, :, 1].unsqueeze(2) - q[:, :, 1].unsqueeze(1))

  return torch.sum(w_1 * (a_1 + b_2)) / 2.0


def o_p_q(r_1, t_2):
        g_1 = r_1[:, :, 0].unsqueeze(2)
        g_2 = r_1[:, :, 1].unsqueeze(2)
        
        h_1 = t_2[:, :, 0].unsqueeze(2)
        h_2 = t_2[:, :, 1].unsqueeze(2)
        
        j_1 = torch.abs(g_1 - g_1.transpose(1, 2))
        j_2 = torch.abs(g_2 - g_2.transpose(1, 2))
        
        k_1 = (h_1 + h_1.transpose(1, 2)) / 2.0
        
        k_2 = (h_2 + h_2.transpose(1, 2)) / 2.0
        
        l_1 = torch.relu(k_1 - j_1)
        l_2 = torch.relu(k_2 - j_2)
        m_0 = 1.0 - torch.eye(r_1.size(1), device=r_1.device).unsqueeze(0)
        
        return torch.sum((l_1 * l_2) * m_0) / 2.0


def main():
    t_t_1 = time.time()
    d_e_v = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    f_d = np.load('./processed_mgc_superblue14.npz', allow_pickle=True)
    d_d = f_d['data']
    c_c = f_d['connectivity']

    n_n = d_d[0]['pos'].shape[0]

    g_g = Generator(n_n, 64).to(d_e_v)
    g_g.load_state_dict(torch.load('cgan_gnn_generator.pth', map_location=d_e_v))
    g_g.eval()

    s_s = torch.tensor(d_d[0]['feat'], dtype=torch.float32, device=d_e_v).unsqueeze(0)

    a_a = torch.tensor(c_c, dtype=torch.float32, device=d_e_v).unsqueeze(0)

    with torch.no_grad():
        z_z = torch.zeros(1, n_n, 64, device=d_e_v)
        p_p = g_g(z_z, s_s, a_a)

        v_1 = x_y_z_1(p_p, a_a)

        v_2 = o_p_q(p_p, s_s)

    t_t_2 = time.time()
    
    print("HPWL:", v_1.item())
    print("Overlap:", v_2.item())
    print("Time:", t_t_2 - t_t_1)

if __name__ == '__main__': 
    main()
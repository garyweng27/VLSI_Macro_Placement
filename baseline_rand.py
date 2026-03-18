import numpy as np
import time

def o_o_1(a_z, b_y):
    s_1 = np.abs(a_z[:, 0:1] - a_z[:, 0:1].T)
    s_2 = np.abs(a_z[:, 1:2] - a_z[:, 1:2].T)
    return np.sum(b_y * (s_1 + s_2)) / 2.0

def p_p_2(c_x, d_w):
    n_9 = c_x.shape[0]
    f_1 = c_x[:, 0:1]
    f_2 = c_x[:, 1:2]
    g_1 = d_w[:, 0:1] / 2.0
    g_2 = d_w[:, 1:2] / 2.0
    j_1 = np.abs(f_1 - f_1.T)
    j_2 = np.abs(f_2 - f_2.T)
    k_1 = g_1 + g_1.T
    k_2 = g_2 + g_2.T
    l_1 = np.maximum(0, k_1 - j_1)
    l_2 = np.maximum(0, k_2 - j_2)
    m_0 = 1.0 - np.eye(n_9)
    return np.sum((l_1 * l_2) * m_0) / 2.0

def x_c_v_b():
    q_1 = np.load('./processed_mgc_superblue12.npz', allow_pickle=True)
    w_2 = q_1['data'][0]['feat']
    e_3 = q_1['connectivity']
    r_4 = w_2.shape[0]
    t_5 = np.random.rand(r_4, 2)
    y_6 = o_o_1(t_5, e_3)
    u_7 = p_p_2(t_5, w_2)
    i_8 = time.time()
    o_9 = 0
    
    while u_7 > 0.0001:
        p_0 = t_5.copy()
        a_1 = np.random.randint(0, r_4)
        p_0[a_1] = p_0[a_1] + (np.random.rand(2) - 0.5) * 0.1
        p_0 = np.clip(p_0, 0, 1)
        
        s_2 = p_p_2(p_0, w_2)
        d_3 = o_o_1(p_0, e_3)
        
        if s_2 < u_7 or (s_2 == u_7 and d_3 < y_6):
            t_5 = p_0
            u_7 = s_2
            y_6 = d_3
            
        o_9 += 1
        if o_9 > 150000:
            break
            
    f_4 = time.time()
    print("HC HPWL:", y_6)
    print("HC Overlap:", u_7)
    print("Time:", f_4 - i_8)
    print("Steps:", o_9)

if __name__ == '__main__':
    x_c_v_b()
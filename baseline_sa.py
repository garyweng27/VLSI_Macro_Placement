import numpy as np
import random
import math
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def pack_sequence_pair(seq_x, seq_y, widths, heights):
    num_macros = len(seq_x)
    x_coords = np.zeros(num_macros)
    y_coords = np.zeros(num_macros)
    pos_x = {macro: idx for idx, macro in enumerate(seq_x)}
    pos_y = {macro: idx for idx, macro in enumerate(seq_y)}
    
    for i in range(num_macros):
        curr_macro = seq_x[i]
        max_x = 0
        for j in range(i):
            prev_macro = seq_x[j]
            if pos_y[prev_macro] < pos_y[curr_macro]:
                if x_coords[prev_macro] + widths[prev_macro] > max_x:
                    max_x = x_coords[prev_macro] + widths[prev_macro]
        x_coords[curr_macro] = max_x
        
    for i in range(num_macros):
        curr_macro = seq_x[i]
        max_y = 0
        for j in range(i):
            prev_macro = seq_x[j]
            if pos_y[prev_macro] > pos_y[curr_macro]:
                if y_coords[prev_macro] + heights[prev_macro] > max_y:
                    max_y = y_coords[prev_macro] + heights[prev_macro]
        y_coords[curr_macro] = max_y
        
    return x_coords, y_coords

def calculate_cost(x_coords, y_coords, widths, heights, adj_matrix):
    centers_x = x_coords + widths / 2.0
    centers_y = y_coords + heights / 2.0
    dx = np.abs(centers_x[:, None] - centers_x[None, :])
    dy = np.abs(centers_y[:, None] - centers_y[None, :])
    hpwl = np.sum(adj_matrix * (dx + dy)) / 2.0
    
    max_w = max(0, np.max(x_coords + widths) - 1.0)
    max_h = max(0, np.max(y_coords + heights) - 1.0)
    
    penalty = (max_w + max_h) * 50000.0
    return hpwl + penalty, hpwl

def plot_layout(x_coords, y_coords, widths, heights, adj_matrix, filename):
    fig, ax = plt.subplots(figsize=(12, 12))
    num_macros = len(widths)
    
    for i in range(num_macros):
        rect = patches.Rectangle(
            (x_coords[i], y_coords[i]),
            widths[i],
            heights[i],
            linewidth=1.5,
            edgecolor='navy',
            facecolor='skyblue',
            alpha=0.6
        )
        ax.add_patch(rect)
        ax.text(
            x_coords[i] + widths[i] / 2.0,
            y_coords[i] + heights[i] / 2.0,
            str(i),
            color='black',
            fontweight='bold',
            fontsize=9,
            ha='center',
            va='center'
        )
        
    centers_x = x_coords + widths / 2.0
    centers_y = y_coords + heights / 2.0
    
    for i in range(num_macros):
        for j in range(i + 1, num_macros):
            if adj_matrix[i, j] > 0:
                ax.plot(
                    [centers_x[i], centers_x[j]],
                    [centers_y[i], centers_y[j]],
                    color='gray',
                    alpha=0.3,
                    linewidth=0.5
                )
                
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Simulated Annealing Layout (mgc_superblue12)', fontsize=16)
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    data_path = './processed_mgc_superblue12.npz'
    dataset = np.load(data_path, allow_pickle=True)
    features = dataset['data'][0]['feat']
    adj_matrix = dataset['connectivity']
    
    num_macros = len(features)
    widths = features[:, 0]
    heights = features[:, 1]
    
    seq_x = list(range(num_macros))
    seq_y = list(range(num_macros))
    random.shuffle(seq_x)
    random.shuffle(seq_y)
    
    best_x, best_y = pack_sequence_pair(seq_x, seq_y, widths, heights)
    best_cost, best_hpwl = calculate_cost(best_x, best_y, widths, heights, adj_matrix)
    
    curr_seq_x = list(seq_x)
    curr_seq_y = list(seq_y)
    curr_cost = best_cost
    
    temperature = 100.0
    step_count = 0
    
    while temperature > 0.1:
        for _ in range(300):
            next_seq_x = list(curr_seq_x)
            next_seq_y = list(curr_seq_y)
            
            action = random.randint(0, 2)
            idx1, idx2 = random.sample(range(num_macros), 2)
            
            if action == 0 or action == 2:
                next_seq_x[idx1], next_seq_x[idx2] = next_seq_x[idx2], next_seq_x[idx1]
            if action == 1 or action == 2:
                next_seq_y[idx1], next_seq_y[idx2] = next_seq_y[idx2], next_seq_y[idx1]
                
            x_coords, y_coords = pack_sequence_pair(next_seq_x, next_seq_y, widths, heights)
            cost, hpwl = calculate_cost(x_coords, y_coords, widths, heights, adj_matrix)
            
            delta_cost = cost - curr_cost
            
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / temperature):
                curr_seq_x = next_seq_x
                curr_seq_y = next_seq_y
                curr_cost = cost
                
                if curr_cost < best_cost:
                    best_cost = curr_cost
                    best_hpwl = hpwl
                    best_x = x_coords
                    best_y = y_coords
                    
            step_count += 1
        temperature *= 0.95
        
    print("SA HPWL:", best_hpwl)
    print("Steps:", step_count)
    plot_layout(best_x, best_y, widths, heights, adj_matrix, 'sa_layout.png')
    print("Image saved as sa_layout.png")

if __name__ == '__main__':
    main()
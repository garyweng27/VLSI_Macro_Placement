import numpy as np
import os
import glob
from collections import defaultdict

# ================= 設定路徑 =================
GRAPH_DIR = './graph_infromation'  
PLACE_DIR = './instance_placement_micron'
DESIGN_NAME = 'mgc_superblue14' 
MACRO_AREA_THRESHOLD = 500.0
# ==========================================

def preprocess():
    print(f"🚀 (V3 最終版) 開始處理設計: {DESIGN_NAME}")

    # 1. 讀取 Node 資訊
    node_path = os.path.join(GRAPH_DIR, 'node_attr', f'{DESIGN_NAME}_node_attr.npy')
    node_attr = np.load(node_path, allow_pickle=True)
    instance_names = node_attr[0] 

    # 2. 篩選 Macro
    print("   🔍 正在篩選 Macro...")
    place_files = glob.glob(os.path.join(PLACE_DIR, f'*{DESIGN_NAME}*.npy'))
    ref_place_data = np.load(place_files[0], allow_pickle=True).item()
    
    global_max_x, global_max_y = 0, 0
    temp_macro_list = []
    
    for idx, name in enumerate(instance_names):
        if name in ref_place_data:
            val = ref_place_data[name]
            width = val[2] - val[0]
            height = val[3] - val[1]
            area = width * height
            
            global_max_x = max(global_max_x, val[2])
            global_max_y = max(global_max_y, val[3])

            if area > MACRO_AREA_THRESHOLD:
                temp_macro_list.append(idx)
    
    macro_indices = np.array(temp_macro_list)
    num_macros = len(macro_indices)
    print(f"   ✅ 篩選出 {num_macros} 個 Macro")
    
    # 建立一個 map: 原始 Node ID -> 我們的 Macro ID (0~3)
    node_id_to_macro_idx = { node_id: i for i, node_id in enumerate(macro_indices) }

    # ==========================================
    # 🔗 新增功能：計算連線權重 (Connectivity)
    # ==========================================
    print("   🔗 正在計算 Macro 連接矩陣 (Connectivity Matrix)...")
    pin_path = os.path.join(GRAPH_DIR, 'pin_attr', f'{DESIGN_NAME}_pin_attr.npy')
    pin_data = np.load(pin_path, allow_pickle=True)
    
    # pin_data[1] 是 Net ID, pin_data[2] 是 Node ID
    net_ids = pin_data[1]
    node_ids = pin_data[2]
    
    # 1. 找出每個 Net 連接了哪些 Macro
    net_to_macros = defaultdict(set)
    
    # 這裡我們只關心連接到 Macro 的 Pin
    # 為了加速，我們只遍歷那些屬於 Macro 的 Node ID
    # 但因為 pin_data 是以 pin 為單位，我們直接遍歷整個 array 比較快
    for i in range(len(net_ids)):
        n_id = node_ids[i]
        if n_id in node_id_to_macro_idx: # 如果這個 pin 屬於我們選出的 4 個 Macro 之一
            macro_idx = node_id_to_macro_idx[n_id]
            net_id = net_ids[i]
            net_to_macros[net_id].add(macro_idx)
            
    # 2. 建立鄰接矩陣 (Adjacency Matrix)
    # 形狀: [num_macros, num_macros]
    adj_matrix = np.zeros((num_macros, num_macros), dtype=np.float32)
    
    for net_id, connected_macros in net_to_macros.items():
        macros = list(connected_macros)
        if len(macros) > 1:
            # 如果這條網線連接了多個 Macro，他們之間就有連線關係
            # 兩兩之間 +1
            for i in range(len(macros)):
                for j in range(i + 1, len(macros)):
                    u, v = macros[i], macros[j]
                    adj_matrix[u, v] += 1
                    adj_matrix[v, u] += 1 # 對稱
                    
    print(f"   📊 連接矩陣預覽:\n{adj_matrix}")
    if adj_matrix.sum() == 0:
        print("   ⚠️ 警告：這 4 個 Macro 之間似乎沒有直接連線 (可能是透過 Std Cells 間接連接)。")

    # ==========================================
    # 📦 打包數據
    # ==========================================
    print("   📦 正在打包數據...")
    processed_data_list = []

    for p_file in place_files:
        place_data = np.load(p_file, allow_pickle=True).item()
        
        current_pos = [] 
        current_feat = [] 
        
        valid_file = True
        for idx in macro_indices:
            name = instance_names[idx]
            if name not in place_data:
                valid_file = False
                break
            
            val = place_data[name]
            width = val[2] - val[0]
            height = val[3] - val[1]
            cx = (val[0] + val[2]) / 2.0
            cy = (val[1] + val[3]) / 2.0
            
            current_pos.append([cx / global_max_x, cy / global_max_y])
            current_feat.append([width / global_max_x, height / global_max_y])
            
        if valid_file:
            processed_data_list.append({
                'pos': np.array(current_pos, dtype=np.float32),
                'feat': np.array(current_feat, dtype=np.float32)
            })

    save_path = f'./processed_{DESIGN_NAME}.npz'
    np.savez(save_path, 
             data=processed_data_list, 
             macro_indices=macro_indices,
             canvas_size=np.array([global_max_x, global_max_y]),
             connectivity=adj_matrix) # <--- 新增存檔
    
    print(f"\n🎉 處理完成！已包含 Connectivity Matrix。")

if __name__ == "__main__":
    preprocess()
import numpy as np
import os
import glob

# ================= 設定路徑 =================
GRAPH_DIR = './graph_infromation'  
PLACE_DIR = './instance_placement_micron'
DESIGN_NAME = 'mgc_pci_bridge32_a' 
# ==========================================

def deep_dive():
    print(f"🕵️‍♂️ 深入分析設計: {DESIGN_NAME}\n")

    # 1. 載入 Node Attr (名字與型號)
    node_path = os.path.join(GRAPH_DIR, 'node_attr', f'{DESIGN_NAME}_node_attr.npy')
    node_attr = np.load(node_path, allow_pickle=True)
    
    # 取出 Cell Types (第二列)
    cell_types = node_attr[1]
    unique_types = np.unique(cell_types)
    
    print(f"📊 元件類型分析 (Cell Types):")
    print(f"   總共有 {len(unique_types)} 種不同的 Cell Type。")
    print(f"   前 20 種範例: {unique_types[:20]}")
    
    # 判斷一下有沒有像 Macro 的東西 (通常不是以 'in01...' 開頭的)
    potential_macros = [t for t in unique_types if not t.startswith('in01') and not t.startswith('DFF')]
    print(f"   🧐 疑似 Macro 的型號 ({len(potential_macros)} 個): {potential_macros[:10]} ...")

    print("-" * 30)

    # 2. 載入 Placement (座標與尺寸?)
    search_pattern = os.path.join(PLACE_DIR, f'*{DESIGN_NAME}*.npy')
    target_place_file = glob.glob(search_pattern)[0]
    
    place_raw = np.load(target_place_file, allow_pickle=True).item() # 解包字典
    
    # 隨便抓一個「疑似 Macro」的元件來看看它的 Placement 數據長怎樣
    # 如果找不到 Macro，就抓第一個元件
    sample_key = None
    
    # 嘗試找一個名字對得上的
    all_keys = list(place_raw.keys())
    sample_key = all_keys[0] # 預設抓第一個
    
    print(f"🗝️ 檢查元件: {sample_key}")
    sample_val = place_raw[sample_key]
    
    print(f"   數據類型: {type(sample_val)}")
    
    if isinstance(sample_val, (np.ndarray, list)):
        sample_val = np.array(sample_val)
        print(f"   數據內容: {sample_val}")
        print(f"   數據長度: {len(sample_val)}")
        
        if len(sample_val) == 2:
            print("   👉 格式看起來是 [x, y]")
        elif len(sample_val) == 4:
            print("   👉 格式看起來是 [x, y, width, height] (這是我們最想要的！)")
        elif len(sample_val) == 8:
            print("   👉 格式看起來是 [x1, y1, x2, y2, ...] (可能是四個角落座標)")
            
    print("-" * 30)

if __name__ == "__main__":
    deep_dive()
import numpy as np
import os
import glob

# ================= 路徑設定 =================
GRAPH_DIR = './graph_infromation'  
PLACE_DIR = './instance_placement_micron'
DESIGN_NAME = 'mgc_pci_bridge32_a' 
# ==========================================

def analyze_area_distribution():
    print(f"📊 正在分析面積分佈: {DESIGN_NAME}")

    # 1. 讀取名字與 Placement
    node_path = os.path.join(GRAPH_DIR, 'node_attr', f'{DESIGN_NAME}_node_attr.npy')
    node_attr = np.load(node_path, allow_pickle=True)
    instance_names = node_attr[0] # 名字
    instance_types = node_attr[1] # 類型 (方便我們對照)

    place_files = glob.glob(os.path.join(PLACE_DIR, f'*{DESIGN_NAME}*.npy'))
    place_data = np.load(place_files[0], allow_pickle=True).item()
    
    # 2. 收集所有元件的面積
    stats = [] # 存 (name, type, width, height, area)
    
    for i, name in enumerate(instance_names):
        if name in place_data:
            val = place_data[name]
            if len(val) == 4:
                w, h = val[2], val[3]
                area = w * h
                stats.append((name, instance_types[i], w, h, area))
    
    # 3. 排序 (由大到小)
    stats.sort(key=lambda x: x[4], reverse=True)
    
    print(f"\n📈 面積最大的前 30 個元件:")
    print(f"{'Index':<6} {'Name':<20} {'Type':<15} {'WxH':<15} {'Area':<10}")
    print("-" * 70)
    for i in range(min(30, len(stats))):
        s = stats[i]
        print(f"{i:<6} {s[0][:18]:<20} {s[1]:<15} {s[2]:.1f}x{s[3]:.1f}     {s[4]:.1f}")
        
    print(f"\n📉 面積最小的前 10 個元件 (檢查 Standard Cell):")
    for i in range(1, 11):
        idx = len(stats) - i
        s = stats[idx]
        print(f"{idx:<6} {s[0][:18]:<20} {s[1]:<15} {s[2]:.1f}x{s[3]:.1f}     {s[4]:.1f}")

if __name__ == "__main__":
    analyze_area_distribution()
import numpy as np
import os
import glob

# ================= 設定路徑 =================
GRAPH_DIR = './graph_infromation'  
DESIGN_NAME = 'mgc_pci_bridge32_a' 
# ==========================================

def inspect_pins():
    print(f"🔍 正在檢查 Pin 資訊: {DESIGN_NAME}")

    # 1. 找到 Pin 檔案
    # 注意：資料夾結構可能是 pin_attr/mgc_..._pin_attr.npy
    pin_path = os.path.join(GRAPH_DIR, 'pin_attr', f'{DESIGN_NAME}_pin_attr.npy')
    
    if not os.path.exists(pin_path):
        print(f"❌ 找不到 Pin 檔案: {pin_path}")
        return

    print(f"✅ 讀取檔案: {os.path.basename(pin_path)}")
    pin_data = np.load(pin_path, allow_pickle=True)
    
    # 2. 分析格式
    print(f"   資料形狀 (Shape): {pin_data.shape}")
    
    # 印出前 5 筆，讓我們看看哪一欄是 Node Name，哪一欄是 Net Name
    print("-" * 50)
    print("   前 5 筆 Pin 資料 (請截圖這部分):")
    print(pin_data[:5])
    print("-" * 50)

    # 3. 嘗試讀取 Node Attr 來對照
    node_path = os.path.join(GRAPH_DIR, 'node_attr', f'{DESIGN_NAME}_node_attr.npy')
    if os.path.exists(node_path):
        node_attr = np.load(node_path, allow_pickle=True)
        # 隨便抓一個 Macro 的名字來搜尋
        # 假設我們之前找到的第一個 Macro 是 'h2a/pci_io_mux_cbe'
        target_macro = 'h2a/pci_io_mux_cbe' 
        print(f"\n   🔍 嘗試在 Pin 資料中搜尋 Macro: {target_macro}")
        
        # 檢查它出現在哪一欄
        rows, cols = np.where(pin_data == target_macro)
        if len(rows) > 0:
            print(f"   🎉 找到了！它出現在 Column {cols[0]} (這是 Node Name 的欄位)")
            print(f"   範例資料: {pin_data[rows[0]]}")
        else:
            print("   ⚠️ 沒找到這個名字，可能 Pin 資料用的是 Index 而不是 Name？")

if __name__ == "__main__":
    inspect_pins()
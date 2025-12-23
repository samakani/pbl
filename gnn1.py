import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import NNConv
from scipy.io import loadmat
import os
import glob
from torch_geometric.nn import global_mean_pool


class StressGNN(torch.nn.Module):
    def __init__(self, node_in, edge_in, hidden_dim):
        super(StressGNN, self).__init__()
        
        # --- 層の定義 ---
        # NNConv: エッジ特徴量(距離)を用いて畳み込みを行う
        
        # 1層目: 入力(node_in) -> 隠れ層(hidden_dim)
        # エッジ特徴量(edge_in)を変換するNN
        nn1 = Sequential(Linear(edge_in, node_in * hidden_dim), ReLU())
        self.conv1 = NNConv(node_in, hidden_dim, nn1, aggr='mean')
        
        # 2層目: 隠れ層 -> 隠れ層
        nn2 = Sequential(Linear(edge_in, hidden_dim * hidden_dim), ReLU())
        self.conv2 = NNConv(hidden_dim, hidden_dim, nn2, aggr='mean')

        # 3層目: 隠れ層 -> 隠れ層 (層を深くして表現力を上げる)
        nn3 = Sequential(Linear(edge_in, hidden_dim * hidden_dim), ReLU())
        self.conv3 = NNConv(hidden_dim, hidden_dim, nn3, aggr='mean')

        # 出力層: 隠れ層 -> 応力(1次元)
        self.out = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Activation関数 (ReLU) を挟みながら伝播
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.relu(self.conv3(x, edge_index, edge_attr))

        # 最終出力
        x = self.out(x)
        return x


# data_cellの構成(すべてlist形式で格納)
# data_cell[0][0]：節点開始点
# data_cell[0][1]：節点終了点
# data_cell[1]　 ：節点座標[[X,Y,Z], ...]
# data_cell[2]　 ：節点間距離[[distance], ...]
# data_cell[3]　 ：各節点の表面有無[[node_id, flag], ...] (0:内部 ,1:表面)
# data_cell[4]　 ：各節点の応力[[node_id, stress], ...] [MPa]
# data_cell[5]　 ：各節点の温度[-](全学習データで正規化した温度、0:MIN温度 ,1:MAX温度)

def load_data_from_file(filepath):
    """Load a single DATA.mat file and convert to PyG Data object"""
    # scipy.ioはnumpy配列として読み込む（mat4pyより10~100倍高速）
    DATA = loadmat(filepath, squeeze_me=True, struct_as_record=False)
    data_cell = DATA["DATA_CELL"]
    
    # scipy.ioの構造: squeeze_me=Trueでアクセスが簡単になる
    # 1. ノード特徴量 (x) の作成
    coords = torch.tensor(data_cell[1], dtype=torch.float)
    
    # 表面有無: data_cell[3]は[node_id, flag]のペアの配列
    surface_data = data_cell[3]
    if len(surface_data.shape) == 1:
        # リストのような構造の場合
        surface = torch.tensor([item[1] for item in surface_data], dtype=torch.float).view(-1, 1)
    else:
        surface = torch.tensor(surface_data[:, 1], dtype=torch.float).view(-1, 1)
    
    x = torch.cat([coords, surface], dim=1)
    
    # 2. エッジインデックス (edge_index) の作成
    # MATLAB uses 1-based indexing, so subtract 1 to convert to 0-based
    edge_data = data_cell[0]
    source_nodes = edge_data[0, :].astype(int) - 1
    target_nodes = edge_data[1, :].astype(int) - 1
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    
    # 3. エッジ特徴量 (edge_attr) の作成
    distance_data = data_cell[2]
    if len(distance_data.shape) == 1:
        edge_attr = torch.tensor(distance_data, dtype=torch.float).view(-1, 1)
    else:
        edge_attr = torch.tensor(distance_data[:, 0], dtype=torch.float).view(-1, 1)
    
    # 4. 正解ラベル (y) の作成
    stress_data = data_cell[4]
    if len(stress_data.shape) == 1:
        y = torch.tensor([item[1] for item in stress_data], dtype=torch.float).view(-1, 1)
    else:
        y = torch.tensor(stress_data[:, 1], dtype=torch.float).view(-1, 1)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

# 1. データ変換実行（キャッシュ機能付き）
cache_file = "processed_data_cache.pt"

if os.path.exists(cache_file):
    print(f"Loading cached data from {cache_file}...")
    data_list = torch.load(cache_file)
    print(f"Loaded {len(data_list)} datasets from cache!")
else:
    print("Loading data from all folders (No1-No19)...")
    data_list = []
    base_path = "/home/nit37414082/pbl/Aisan/unzip_data"

    # Load data from No1 to No19
    for i in range(1, 20):
        folder = f"No{i}"
        folder_path = os.path.join(base_path, folder)
        
        if not os.path.exists(folder_path):
            print(f"Warning: {folder} does not exist, skipping...")
            continue
        
        # Find all DATA*.mat files in the folder
        mat_files = sorted(glob.glob(os.path.join(folder_path, "DATA*.mat")))
        
        for mat_file in mat_files:
            try:
                data = load_data_from_file(mat_file)
                data_list.append(data)
                print(f"Loaded {os.path.basename(mat_file)} from {folder}: {data.num_nodes} nodes, {data.num_edges} edges")
            except Exception as e:
                print(f"Error loading {mat_file}: {e}")
                import traceback
                traceback.print_exc()  # 詳細なエラー情報を表示

    print(f"\nTotal datasets loaded: {len(data_list)}")
    
    # キャッシュに保存
    print(f"Saving cache to {cache_file}...")
    torch.save(data_list, cache_file)
    print("Cache saved!")

if len(data_list) == 0:
    raise RuntimeError("No data files were loaded!")

# DataLoaderを使用してバッチ処理 (複数のグラフを同時に処理)
# batch_size: 同時に処理するグラフ数
# shuffle: エポックごとにデータをシャッフル
# num_workers: 並列ロードのワーカー数 (0=メインプロセスのみ)
batch_size = 4  # メモリに応じて調整可能
train_loader = DataLoader(data_list, batch_size=batch_size, shuffle=True, num_workers=2)
print(f"DataLoader created: {len(train_loader)} batches of size {batch_size}")

# 2. デバイス設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
model = StressGNN(node_in=4, edge_in=1, hidden_dim=32).to(device)

# 3. 最適化手法と損失関数
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = torch.nn.MSELoss() # 回帰なのでMSE（平均二乗誤差）

# 4. 学習ループ
model.train()
print("\nStarting Training...")

epochs = 50
for epoch in range(epochs):
    total_loss = 0
    num_batches = 0
    
    # DataLoaderからバッチごとにデータを取得
    for batch_data in train_loader:
        # データをデバイスに移動
        batch_data = batch_data.to(device)
        
        optimizer.zero_grad()
        
        # 推論
        out = model(batch_data)
        
        # 損失計算 (out:予測値, batch_data.y:正解応力)
        loss = criterion(out, batch_data.y)
        
        # バックプロパゲーション
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch:04d}: Avg Loss (MSE) = {avg_loss:.4f}")

# 5. 結果確認 (最初のデータセットの最初の5節点分)
model.eval()
with torch.no_grad():
    first_data = data_list[0].to(device)
    pred = model(first_data)
    print("\n--- Prediction Results (First dataset, first 5 nodes) ---")
    print("   True [MPa]   |   Pred [MPa]   |   Diff")
    print("-" * 45)
    for i in range(5):
        true_val = first_data.y[i].item()
        pred_val = pred[i].item()
        print(f" {true_val:12.4f}   | {pred_val:12.4f}   | {abs(true_val - pred_val):.4f}")

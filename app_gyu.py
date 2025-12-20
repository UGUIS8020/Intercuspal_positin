import os
import sys
import argparse
import numpy as np
import trimesh
from tkinter import Tk, filedialog
from scipy.spatial.transform import Rotation as R

# GPU加速の設定
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✓ CuPy GPU加速が利用可能です")
except ImportError:
    print("⚠ CuPy が見つかりません。CPU版で動作します。")
    import numpy as cp  # fallback to numpy
    GPU_AVAILABLE = False

# NOTE:
# 距離計算は gpu_min_distances() を使用する（追加依存: cuVS/pylibraft が必要になる実装は避ける）

def array_to_gpu(arr):
    """numpy array をGPUに転送"""
    if GPU_AVAILABLE and hasattr(cp, 'asarray'):
        return cp.asarray(arr)
    return arr

def array_to_cpu(arr):
    """GPU array をCPUに戻す"""
    if GPU_AVAILABLE and hasattr(arr, 'get'):
        return arr.get()
    return arr


# =============================
# 深噛み検出のロバスト化
# =============================

def compute_deep_metrics(distances, k_ratio=0.005, k_min=6):
    """
    距離配列から複数のロバストな指標を計算する。
    
    Parameters
    ----------
    distances : np.ndarray
        距離配列
    k_ratio : float
        下位点選択の割合（デフォルト0.5%）
    k_min : int
        最小選択点数（デフォルト6点）
    
    Returns
    -------
    dict
        min_abs: 絶対最小値（参考用）
        min_p1: 下位1%点
        deep_guard: 下位k点の中央値（ロバストな深噛み判定値）
        k: 実際に使用した下位点数
    
    Notes
    -----
    - （追加依存を避けた距離計算：gpu_min_distances）
    """
    n = len(distances)
    k = max(k_min, int(n * k_ratio))
    k = min(k, n)  # 配列サイズを超えないように
    
    # 下位k点を取得（O(n)）
    lower_k = np.partition(distances, k-1)[:k]
    
    return {
        'min_abs': float(np.min(distances)),
        'min_p1': float(np.percentile(distances, 1.0)),
        'deep_guard': float(np.median(lower_k)),
        'k': k
    }


def gpu_min_distances(points_a, points_b, batch_a=256, block_b=8192):
    """
    GPU加速版の最小距離計算（点群AからBへの最短距離）
    
    Parameters
    ----------
    points_a : cp.ndarray or np.ndarray
        形状 (N, 3) のクエリ点群
    points_b : cp.ndarray or np.ndarray
        形状 (M, 3) のターゲット点群
    batch_a : int
        Aのバッチサイズ
    block_b : int
        Bのブロックサイズ
    
    Returns
    -------
    cp.ndarray
        各点の最小距離 (N,)
    
    Notes
    -----
    - 追加依存（cuVS/pylibraft）を避けた GPU 距離最小化（gpu_min_distances）
    - メモリ効率的なバッチ処理実装
    """
    if not GPU_AVAILABLE:
        # CPU fallback
        from scipy.spatial.distance import cdist
        dists = cdist(points_a, points_b, metric='euclidean')
        return np.min(dists, axis=1)
    
    points_a_gpu = array_to_gpu(points_a)
    points_b_gpu = array_to_gpu(points_b)
    
    n_a = points_a_gpu.shape[0]
    n_b = points_b_gpu.shape[0]
    
    min_dists = cp.full(n_a, cp.inf, dtype=cp.float32)
    
    for i_a in range(0, n_a, batch_a):
        end_a = min(i_a + batch_a, n_a)
        batch_points = points_a_gpu[i_a:end_a]
        
        batch_min = cp.full(end_a - i_a, cp.inf, dtype=cp.float32)
        
        for j_b in range(0, n_b, block_b):
            end_b = min(j_b + block_b, n_b)
            block_points = points_b_gpu[j_b:end_b]
            
            # ユークリッド距離計算: ||a - b||
            diff = batch_points[:, cp.newaxis, :] - block_points[cp.newaxis, :, :]
            dists = cp.sqrt(cp.sum(diff ** 2, axis=2))
            
            block_min = cp.min(dists, axis=1)
            batch_min = cp.minimum(batch_min, block_min)
        
        min_dists[i_a:end_a] = batch_min
    
    return min_dists


# =============================
# ユーティリティ
# =============================

def select_moving_jaw():
    """どちらの顎を動かすか選択するダイアログ"""
    from tkinter import messagebox
    
    root = Tk()
    root.withdraw()
    
    # ダイアログで選択
    result = messagebox.askyesnocancel(
        "顎の選択",
        "どちらの顎を動かしますか？\n\n「はい」= 下顎を動かす（上顎固定）\n「いいえ」= 上顎を動かす（下顎固定）",
        icon='question'
    )
    
    root.destroy()
    
    if result is None:  # キャンセル
        print("❌ キャンセルされました")
        sys.exit(0)
    elif result:  # はい
        print("✓ 選択: 下顎を動かす（上顎固定）")
        return "lower"
    else:  # いいえ
        print("✓ 選択: 上顎を動かす（下顎固定）")
        return "upper"

def select_two_stl_files():
    """
    ファイルダイアログから STL ファイルを1顎ずつ選択
    1回目: 上顎, 2回目: 下顎
    ※ キャンセル対策: 再試行ループ + topmost
    """
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)  # ★ Windows対策: 最前面表示
    root.update()
    
    print("\n=== STLファイル選択 ===")
    
    # 上顎選択（再試行ループ）
    upper_path = None
    for attempt in range(3):  # 最大3回試行
        print(f"\nまず上顎のSTLファイルを選択してください... (試行 {attempt+1}/3)")
        upper_path = filedialog.askopenfilename(
            title="🦷 上顎（Upper Jaw）のSTLファイルを選択 - STEP 1/2",
            filetypes=[("STL files", "*.stl"), ("All files", "*.*")],
            parent=root
        )
        if upper_path:
            break
        print("⚠️  ファイルが選択されませんでした。もう一度お試しください。")
    
    if not upper_path:
        print("❌ エラー: 上顎 STL が選択されませんでした（3回試行）。")
        root.destroy()
        sys.exit(1)
    
    print(f"✓ 上顎選択完了: {os.path.basename(upper_path)}")
    
    # 下顎選択（再試行ループ）
    lower_path = None
    for attempt in range(3):  # 最大3回試行
        print(f"\n次に下顎のSTLファイルを選択してください... (試行 {attempt+1}/3)")
        lower_path = filedialog.askopenfilename(
            title="🦷 下顎（Lower Jaw）のSTLファイルを選択 - STEP 2/2",
            filetypes=[("STL files", "*.stl"), ("All files", "*.*")],
            parent=root
        )
        if lower_path:
            break
        print("⚠️  ファイルが選択されませんでした。もう一度お試しください。")
    
    if not lower_path:
        print("❌ エラー: 下顎 STL が選択されませんでした（3回試行）。")
        root.destroy()
        sys.exit(1)

    root.update()
    root.destroy()

    if os.path.abspath(upper_path) == os.path.abspath(lower_path):
        print("❌ エラー: 同じ STL が2回選択されています。上顎と下顎は別ファイルを選んでください。")
        sys.exit(1)

    print(f"✓ 下顎選択完了: {os.path.basename(lower_path)}")
    print(f"\n📁 選択されたファイル:")
    print(f"   上顎: {upper_path}")
    print(f"   下顎: {lower_path}")
    print("=" * 50)
    return upper_path, lower_path

def load_mesh_safely(filepath):
    """trimesh で STL を読み込む（簡易チェック付き）"""
    try:
        mesh = trimesh.load(filepath)
        
        # 水密チェック
        is_watertight = mesh.is_watertight
        if not is_watertight:
            print(f"\n{'='*70}")
            print(f"⚠️  重要警告: {os.path.basename(filepath)} は水密ではありません")
            print(f"{'='*70}")
            print(f"\n【影響】")
            print(f"  • 接触面積の推定精度が低下")
            print(f"  • min_dist_raw が異常値（0に寄る/飛ぶ）になる可能性")
            print(f"  • 接触点数・バランス評価の再現性が低下")
            print(f"\n【推奨修復手順（MeshLab）】")
            print(f"  1. MeshLabでSTLを開く")
            print(f"  2. Filters → Cleaning and Repairing → Fill Holes")
            print(f"  3. Filters → Cleaning and Repairing → Remove Non-Manifold Edges")
            print(f"  4. Filters → Cleaning and Repairing → Remove Duplicate Faces")
            print(f"  5. Filters → Cleaning and Repairing → Remove Zero Area Faces")
            print(f"  6. File → Export Mesh As... で上書き保存")
            print(f"\n【注意】本プログラムは継続しますが、結果の信頼性に注意してください")
            print(f"{'='*70}\n")
        
        if len(mesh.vertices) < 100:
            raise ValueError(f"頂点数が少なすぎます: {len(mesh.vertices)}")
        
        status = "✓" if is_watertight else "⚠"
        watertight_str = "水密" if is_watertight else "非水密"
        print(f"{status} {os.path.basename(filepath)} 読み込み ({len(mesh.vertices)} 頂点, {watertight_str})")
        
        return mesh
    except Exception as e:
        print(f"エラー: {filepath} の読み込みに失敗しました")
        print("詳細:", e)
        sys.exit(1)


def per_vertex_area(mesh: trimesh.Trimesh):
    """
    各三角形の面積を3頂点に等分配して頂点面積とする
    （ベクトル化版：高速）
    """
    areas = np.zeros(len(mesh.vertices), dtype=np.float64)
    a3 = (mesh.area_faces / 3.0).astype(np.float64)
    f = mesh.faces
    np.add.at(areas, f[:, 0], a3)
    np.add.at(areas, f[:, 1], a3)
    np.add.at(areas, f[:, 2], a3)
    return areas


def export_contact_points_ply(
    contact_points,
    region_labels,
    output_path,
    region_colors=None
):
    """
    接触点をPLY形式で出力（5ブロック別に色分け）
    
    Parameters
    ----------
    contact_points : np.ndarray (N, 3)
        接触点座標
    region_labels : list of str
        各点の所属ブロック名 ["M_L", "M_R", ...]
    output_path : str
        出力先PLYファイルパス
    region_colors : dict, optional
        ブロック名 → RGB色（0-255）のマッピング
    """
    if region_colors is None:
        region_colors = {
            "M_L": (255, 100, 100),    # 赤系（左大臼歯）
            "M_R": (100, 100, 255),    # 青系（右大臼歯）
            "PM_L": (255, 200, 100),   # オレンジ系（左小臼歯）
            "PM_R": (100, 200, 255),   # 水色系（右小臼歯）
            "ANT": (100, 255, 100),    # 緑系（前歯）
        }
    
    with open(output_path, 'w') as f:
        # PLYヘッダー
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(contact_points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # 頂点データ
        for point, label in zip(contact_points, region_labels):
            color = region_colors.get(label, (128, 128, 128))
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                   f"{color[0]} {color[1]} {color[2]}\n")
    
    print(f"✓ 接触点PLY出力: {output_path} ({len(contact_points)}点)")


# =============================
# 変換行列構築（評価と出力で完全一致させるため）
# =============================

def build_transform_matrix(tx, ty, rx_rad, ry_rad, tz, pivot=None):
    """
    剛体変換行列を構築（4×4のホモジニアス変換）
    
    変換順序: T(pivot) @ R @ T(-pivot) @ T(translation)
    - pivot周りで回転
    - その後、平行移動
    
    Args:
        tx, ty, tz: 平行移動 (mm)
        rx_rad, ry_rad: 回転角（ラジアン、X軸・Y軸）
        pivot: 回転中心 (3,) array、Noneなら原点
    
    Returns:
        4×4 numpy array（ホモジニアス変換行列）
    """
    from scipy.spatial.transform import Rotation as R
    
    # 回転行列（3×3）
    rot_matrix = R.from_euler("xyz", [rx_rad, ry_rad, 0.0]).as_matrix()
    
    # 平行移動ベクトル
    translation = np.array([tx, ty, tz])
    
    if pivot is None:
        pivot = np.zeros(3)
    
    # 4×4 ホモジニアス行列の構築
    # T(translation) @ T(pivot) @ R @ T(-pivot)
    T_neg_pivot = np.eye(4)
    T_neg_pivot[:3, 3] = -pivot
    
    R_mat = np.eye(4)
    R_mat[:3, :3] = rot_matrix
    
    T_pivot = np.eye(4)
    T_pivot[:3, 3] = pivot
    
    T_trans = np.eye(4)
    T_trans[:3, 3] = translation
    
    # 順番: T(translation) @ T(pivot) @ R @ T(-pivot)
    A = T_trans @ T_pivot @ R_mat @ T_neg_pivot
    
    return A


def apply_transform_to_points(points, transform_matrix):
    """
    4×4変換行列を点群に適用
    
    Args:
        points: (N, 3) numpy array
        transform_matrix: (4, 4) numpy array
    
    Returns:
        transformed_points: (N, 3) numpy array
    """
    # ホモジニアス座標に変換 (N, 4)
    N = points.shape[0]
    points_homogeneous = np.hstack([points, np.ones((N, 1))])
    
    # 変換適用
    transformed_homogeneous = (transform_matrix @ points_homogeneous.T).T
    
    # 3D座標に戻す
    return transformed_homogeneous[:, :3]


# =============================
# スコアリング（5本の輪ゴムスプリングモデル）
# =============================

class SpringOcclusionScorer:
    """
    上下歯列を「輪ゴム5本」で引っ張り合うイメージで評価するスコア計算クラス

    5本のバネ:
      - M_L : 左大臼歯ブロック
      - M_R : 右大臼歯ブロック
      - PM_L: 左小臼歯ブロック
      - PM_R: 右小臼歯ブロック
      - ANT : 前歯ブロック（左右まとめて）
    """

    def __init__(
        self,
        upper_mesh: trimesh.Trimesh,
        lower_sample_vertices: np.ndarray,
        lower_sample_areas: np.ndarray,
        contact_threshold: float = 0.03,
        rot_penalty: float = 1.5,
        trans_penalty: float = 2.0,
        moving_jaw: str = "lower",  # "lower" or "upper"
        lower_mesh_for_springs: trimesh.Trimesh = None,  # スプリング配置用（常に下顎）
        pivot: np.ndarray = None,  # 回転中心（重要：evaluate()と出力を一致させる）
    ):
        # 動かす顎を設定（内部的には固定側=upper、動かす側=v0）
        self.moving_jaw = moving_jaw
        self.upper = upper_mesh  # 固定側（上顎または下顎）
        self.v0 = lower_sample_vertices  # 動かす側のサンプル（下顎または上顎）
        self.areas = lower_sample_areas
        
        # 回転中心の設定（evaluate()とSTL出力で同じ変換を使うため）
        self.pivot = np.array(pivot, dtype=np.float32) if pivot is not None else np.zeros(3, dtype=np.float32)
        
        if moving_jaw == "lower":
            print(f"🦷 動作モード: 上顎固定 / 下顎を移動")
        else:
            print(f"🦷 動作モード: 下顎固定 / 上顎を移動")
        
        print(f"🎯 回転中心（pivot）: [{self.pivot[0]:.3f}, {self.pivot[1]:.3f}, {self.pivot[2]:.3f}]")
        
        # ★ メッシュ水密情報を保存（深噛み閾値調整に使用）
        self.mesh_is_watertight = upper_mesh.is_watertight
        if not self.mesh_is_watertight:
            print(f"  ⚠️  非水密STL検知: 深噛み閾値を緩和（測定誤差考慮）")
            print(f"      critical: 0.005mm → 0.010mm, warning: 0.010mm → 0.015mm, caution: 0.015mm → 0.020mm")
            
        self.contact_threshold = contact_threshold
        self.rot_penalty = rot_penalty
        self.trans_penalty = trans_penalty
        
        # GPU加速用データの準備
        if GPU_AVAILABLE:
            # メモリ効率を考慮してfloat32使用
            self.v0_gpu = array_to_gpu(self.v0.astype(np.float32))
            self.areas_gpu = array_to_gpu(self.areas.astype(np.float32))
            self.upper_vertices_gpu = array_to_gpu(upper_mesh.vertices.astype(np.float32))
            self.pivot_gpu = array_to_gpu(self.pivot)
            
            # GPUメモリ使用量を表示
            gpu_memory_mb = (
                self.v0_gpu.nbytes + self.areas_gpu.nbytes + 
                self.upper_vertices_gpu.nbytes
            ) / (1024 * 1024)
            
            print(f"✓ GPU メモリに転送完了: {len(self.v0)} 下顎頂点, {len(upper_mesh.vertices)} 上顎頂点")
            print(f"✓ GPU メモリ使用量: {gpu_memory_mb:.1f} MB")
            print(f"⚠️  注意: GPU距離は上顎【頂点】への最近接（CPU=三角形面への最近接と異なる）")
            
            # メモリ使用量チェック
            if hasattr(cp, 'get_default_memory_pool'):
                mempool = cp.get_default_memory_pool()
                print(f"✓ GPU メモリプール: {mempool.used_bytes()/(1024*1024):.1f} MB 使用中")
        else:
            self.v0_gpu = self.v0
            self.areas_gpu = self.areas
        
        # GPUバイアス補正（後で診断結果で設定される）
        self.gpu_bias = 0.0
        self.use_cpu_final_eval = True  # CPU最終評価フラグ
        
        # 対策B: 探索時の閾値緩和（後で診断結果で設定される）
        self.contact_threshold_search = contact_threshold  # 探索用（後で緩める）
        self.contact_threshold_final = contact_threshold   # 最終確定用（厳密）
        self.search_mode = False  # True=探索モード（緩い閾値）, False=確定モード（厳密閾値）
        
        # 対策: 接触可能性フラグ（絶対当たらない歯を除外）
        self.infeasible_regions = set()  # 接触不可能なブロック名のセット

        
        # ----------------------------
        # 5ブロックへの自動分割
        # ★重要: 領域の境界は下顎基準で定義、マスクはサンプル頂点に適用
        # ----------------------------
        if lower_mesh_for_springs is not None:
            # 上顎を動かす場合：下顎メッシュから境界値（x_mid, y_cut）を計算
            ref_vertices = lower_mesh_for_springs.vertices
            print(f"🎯 スプリング配置: 下顎基準（下顎の座標系で領域境界を定義）")
        else:
            # 下顎を動かす場合：サンプル頂点から境界値を計算
            ref_vertices = self.v0
            print(f"🎯 スプリング配置: 動かす側基準（サンプル頂点から定義）")
        
        # 参照メッシュから境界値を計算
        x_ref = ref_vertices[:, 0]
        y_ref = ref_vertices[:, 1]

        self.x_mid = float(np.median(x_ref))
        y_min, y_max = float(y_ref.min()), float(y_ref.max())
        if y_max == y_min:
            # 万一全て同じ値なら、全部「臼歯」として扱う
            y_cut1 = y_min - 0.1
            y_cut2 = y_min + 0.1
        else:
            dy = y_max - y_min
            y_cut1 = y_min + dy / 3.0        # 大臼歯 / 小臼歯の境
            y_cut2 = y_min + dy * 2.0 / 3.0  # 小臼歯 / 前歯の境

        # サンプル頂点（動かす側）に境界値を適用してマスク作成
        x = self.v0[:, 0]
        y = self.v0[:, 1]
        
        is_left = x <= self.x_mid
        is_right = ~is_left

        band_molar = y <= y_cut1
        band_premolar = (y > y_cut1) & (y <= y_cut2)
        band_ant = y > y_cut2

        mask_M_L = is_left & band_molar
        mask_M_R = is_right & band_molar
        mask_PM_L = is_left & band_premolar
        mask_PM_R = is_right & band_premolar
        mask_ANT = band_ant  # 前歯は左右まとめて一本のゴム

        self.region_masks = {
            "M_L": mask_M_L,
            "M_R": mask_M_R,
            "PM_L": mask_PM_L,
            "PM_R": mask_PM_R,
            "ANT": mask_ANT,
        }
        
        # GPU高速化：region maskをGPUに事前転送（毎回転送しない）
        if GPU_AVAILABLE:
            self.region_masks_gpu = {
                name: cp.asarray(mask) for name, mask in self.region_masks.items()
            }
            print(f"✓ GPU: region masks事前転送完了（5ブロック）")

        # 実際に頂点が存在するブロックだけを「有効バネ」とみなす
        self.valid_regions = [
            name for name, m in self.region_masks.items() if np.any(m)
        ]

        print("\n[ブロック分割（輪ゴム5本）]")
        total_points = len(lower_sample_vertices)
        for name in ["M_L", "M_R", "PM_L", "PM_R", "ANT"]:
            cnt = int(self.region_masks[name].sum())
            pct = cnt / total_points * 100 if total_points > 0 else 0.0
            flag = "✓" if name in self.valid_regions else "（頂点なし）"
            print(f"  {name:5s}: {cnt:4d} 点 ({pct:5.1f}%) {flag}")
        print(f"  有効バネ本数: {len(self.valid_regions)}")
        
        # ★ 左右バランス診断
        M_L_cnt = int(self.region_masks["M_L"].sum())
        M_R_cnt = int(self.region_masks["M_R"].sum())
        PM_L_cnt = int(self.region_masks["PM_L"].sum())
        PM_R_cnt = int(self.region_masks["PM_R"].sum())
        
        if M_L_cnt + M_R_cnt > 0:
            M_ratio = M_L_cnt / (M_L_cnt + M_R_cnt)
            print(f"\n  📊 大臼歯（M）左右比: L={M_L_cnt} vs R={M_R_cnt} → L_ratio={M_ratio:.3f}")
            if abs(M_ratio - 0.5) > 0.15:  # 15%以上偏り
                bias_side = "左" if M_ratio > 0.5 else "右"
                print(f"     ⚠️  大臼歯が{bias_side}に偏っています（分割境界の要確認）")
        
        if PM_L_cnt + PM_R_cnt > 0:
            PM_ratio = PM_L_cnt / (PM_L_cnt + PM_R_cnt)
            print(f"  📊 小臼歯（PM）左右比: L={PM_L_cnt} vs R={PM_R_cnt} → L_ratio={PM_ratio:.3f}")
            if abs(PM_ratio - 0.5) > 0.15:
                bias_side = "左" if PM_ratio > 0.5 else "右"
                print(f"     ⚠️  小臼歯が{bias_side}に偏っています（分割境界の要確認）")

        eps = 1e-12
        self.region_cap = {}
        for name, mask in self.region_masks.items():
            cap = float(self.areas[mask].sum()) if np.any(mask) else 0.0
            self.region_cap[name] = cap

        capL = self.region_cap["M_L"] + self.region_cap["PM_L"]
        capR = self.region_cap["M_R"] + self.region_cap["PM_R"]
        self.target_L_ratio = capL / (capL + capR + eps)

        # 左側の中で PM_L が占める"自然な比率"（欠損でM_Lが少ないとここが上がる）
        self.target_PM_L_share = self.region_cap["PM_L"] / (capL + eps)
    def __del__(self):
        """GPUメモリを適切にクリーンアップ"""
        if GPU_AVAILABLE and hasattr(self, 'v0_gpu'):
            # 明示的なメモリ解放は CuPy が自動でやってくれるが、
            # 大きなデータの場合は手動でもできる
            if hasattr(cp, 'get_default_memory_pool'):
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
    def _gpu_nearest_distances(self, transformed_vertices_gpu):
        """GPU加速版最近接距離計算（メモリ効率改善版：上顎もブロック分割）"""
        if not GPU_AVAILABLE:
            # CPU fallback
            transformed_cpu = array_to_cpu(transformed_vertices_gpu)
            _, distances, _ = self.upper.nearest.on_surface(transformed_cpu)
            return np.asarray(distances)
        
        # 🔧 修正2: メモリ爆発を防ぐため上顎もブロック分割して最小値更新
        n_lower = transformed_vertices_gpu.shape[0]
        n_upper = self.upper_vertices_gpu.shape[0]
        
        distances = cp.zeros(n_lower, dtype=cp.float32)
        
        # 適応的バッチサイズ（下顎側）
        batch_size = min(256, n_lower)  # メモリ安全な256に調整
        upper_block = 8192  # 上顎ブロックサイズ（4096〜16384で調整可能）
        
        if not hasattr(self, '_gpu_mode_notified'):
            print(f"🚀 GPU高速並列計算（メモリ最適化版）: 下顎バッチ={batch_size}, 上顎ブロック={upper_block}")
            self._gpu_mode_notified = True
        
        for i in range(0, n_lower, batch_size):
            end_i = min(i + batch_size, n_lower)
            batch_lower = transformed_vertices_gpu[i:end_i]  # (B, 3)
            
            # 各下顎点の最小距離の二乗を追跡
            min_dist_sq = cp.full((end_i - i,), cp.inf, dtype=cp.float32)
            
            # 上顎をブロック分割して最小値を更新
            for j in range(0, n_upper, upper_block):
                end_j = min(j + upper_block, n_upper)
                upper_block_vertices = self.upper_vertices_gpu[j:end_j]  # (U, 3)
                
                # Broadcasting: (B, 1, 3) - (1, U, 3) = (B, U, 3)
                diff = batch_lower[:, None, :] - upper_block_vertices[None, :, :]
                dist_sq = cp.sum(diff * diff, axis=2)  # (B, U)
                
                # 最小値を更新
                min_dist_sq = cp.minimum(min_dist_sq, cp.min(dist_sq, axis=1))
            
            # 平方根を取って距離に変換
            distances[i:end_i] = cp.sqrt(min_dist_sq)
        
        return distances

    def evaluate_batch(self, poses_array, max_dist_clip=0.05):
        """
        複数の姿勢を同時にGPUで評価（超高速化）
        poses_array: (N, 4) の配列 [[tx1, rx1, ry1, tz1], [tx2, rx2, ry2, tz2], ...]
        戻り値: scores配列, infos配列
        """
        if not GPU_AVAILABLE:
            # CPU fallback: 単発評価を繰り返す
            results = [self.evaluate(pose[0], pose[1], pose[2], pose[3], max_dist_clip) 
                      for pose in poses_array]
            scores = [r[0] for r in results]
            infos = [r[1] for r in results]
            return scores, infos
        
        n_poses = len(poses_array)
        scores = []
        infos = []
        
        if not hasattr(self, '_batch_mode_notified'):
            print(f"⚡ GPU並列バッチ評価: {n_poses} 姿勢")
            self._batch_mode_notified = True
        
        # 高速バッチ処理
        batch_size = min(50, n_poses)  # より大きなバッチで高速化
        
        for batch_start in range(0, n_poses, batch_size):
            batch_end = min(batch_start + batch_size, n_poses)
            batch_poses = poses_array[batch_start:batch_end]
            
            for pose in batch_poses:
                # 個別評価を高速化版で実行
                score, info = self.evaluate(pose[0], pose[1], pose[2], pose[3], max_dist_clip)
                scores.append(score)
                infos.append(info)
        
        return scores, infos

    def region_gap_info(self, tx, rx_rad, ry_rad, tz, near_th=0.10):
        """
        現在姿勢での「各ブロックの最近接距離(min)」と
        「near_th以内に入っている点数(near_count)」を返す。
        ※ evaluate() と同じ pivot 回り変換で整合させる（重要）
        """
        ty = 0.0
        rot = R.from_euler("xyz", [rx_rad, ry_rad, 0.0]).as_matrix()
        
        # ★ pivot 回り回転で evaluate() と完全一致
        p = self.pivot
        transformed = (rot @ (self.v0 - p).T).T + p + np.array([tx, ty, tz])

        _, distances, _ = self.upper.nearest.on_surface(transformed)
        distances = np.asarray(distances)

        out = {}
        for name, mask in self.region_masks.items():
            if not np.any(mask):
                out[name] = {"min": float("inf"), "near_count": 0}
                continue
            d = distances[mask]
            out[name] = {
            "min": float(d.min()),
            "p10": float(np.percentile(d, 10)),
            "p25": float(np.percentile(d, 25)),  # 四分位点追加で分布把握強化
            "near_count": int(np.sum(d <= near_th)),
            "contact_potential": float(np.sum(d <= self.contact_threshold))  # 接触ポテンシャル
            }
        return out

    def update_feasibility(self, tx_range=(-0.5, 0.5), tz_range=(-1.0, 2.0), sample_points=9):
        """
        探索範囲内で接触可能性を判定し、絶対当たらないブロックを特定
        """
        print("\n🔍 接触可能性診断（絶対当たらない歯の検出）:")
        
        # 探索範囲の代表点でテスト
        tx_vals = np.linspace(tx_range[0], tx_range[1], 3)
        tz_vals = np.linspace(tz_range[0], tz_range[1], 3)
        
        region_min_dists = {name: [] for name in self.region_masks.keys()}
        
        for tx in tx_vals:
            for tz in tz_vals:
                gap_info = self.region_gap_info(tx, 0.0, 0.0, tz, near_th=0.20)
                for name, info in gap_info.items():
                    region_min_dists[name].append(info["min"])
        
        # 各ブロックの接触可能性を判定
        self.infeasible_regions.clear()
        for name, min_dists in region_min_dists.items():
            overall_min = min(min_dists)
            near_count_max = max([
                self.region_gap_info(tx, 0.0, 0.0, tz, near_th=0.20)[name]["near_count"]
                for tx in tx_vals for tz in tz_vals
            ])
            
            # 判定基準: 最短距離>0.30mm かつ 0.20mm以内の点が0個
            if overall_min > 0.30 and near_count_max == 0:
                self.infeasible_regions.add(name)
                print(f"  {name}: INFEASIBLE (min={overall_min:.3f}mm, near=0) → 探索から除外")
            else:
                print(f"  {name}: feasible (min={overall_min:.3f}mm, near={near_count_max})")
        
        if self.infeasible_regions:
            print(f"✓ {len(self.infeasible_regions)}個のブロックを探索から除外: {list(self.infeasible_regions)}")
        else:
            print("✓ 全ブロックが接触可能範囲内")

    # ----------------------------
    # 姿勢評価
    # ----------------------------

    def evaluate(self, tx, rx_rad, ry_rad, tz, max_dist_clip=0.05, force_cpu=False):
        """
        姿勢 (tx, rx, ry, tz) に対するスコアを返す（GPU加速版 + CPU最終評価オプション）
        - tx: 左右方向スライド（mm）
        - rx, ry: ラジアン（X, Y軸まわりの回転）
        - tz: 垂直方向（mm）
        - force_cpu: True なら CPU で確定評価
        ty は 0 固定（前後スライドはここでは見ない）

        戻り値:
          score, info_dict
        """
        ty = 0.0

        # CPU最終評価が有効で force_cpu=True なら CPU で実行
        if force_cpu or (not GPU_AVAILABLE):
            rot = R.from_euler("xyz", [rx_rad, ry_rad, 0.0]).as_matrix()
            # ★重要：pivot回りで回転してから平行移動
            # v' = R @ (v - pivot) + pivot + t
            p = self.pivot
            transformed = (rot @ (self.v0 - p).T).T + p + np.array([tx, ty, tz])
            _, distances, _ = self.upper.nearest.on_surface(transformed)
            dist_raw = np.asarray(distances)  # 生距離
            min_dist_raw = float(dist_raw.min())
            
            # 🔍 距離分布診断（初回のみ表示）
            if not hasattr(self, '_dist_diagnosed'):
                p_percentile = np.percentile(dist_raw, [0, 1, 5, 10, 50, 90, 95, 99, 100])
                print(f"🔍 距離分布(mm): {p_percentile}")
                print(f"   <=0.035mm: {np.mean(dist_raw <= 0.035):.3f}（{np.sum(dist_raw <= 0.035)}点）")
                print(f"   <=0.050mm: {np.mean(dist_raw <= 0.050):.3f}（{np.sum(dist_raw <= 0.050)}点）")
                print(f"   0.035-0.050mm帯: {np.sum((dist_raw > 0.035) & (dist_raw <= 0.050))}点（ニアミス領域）")
                self._dist_diagnosed = True
            
            # ✅ Step0修正: 接触判定は生距離で行う（クリップ前）
            current_threshold = self.contact_threshold_search if self.search_mode else self.contact_threshold_final
            contact_mask_gpu = dist_raw <= current_threshold  # ★判定は生距離
            
            d_gpu = np.clip(dist_raw, 0.0, max_dist_clip)    # ★クリップは重み計算用
            
            if force_cpu and not hasattr(self, '_cpu_final_notified'):
                print("🎯 CPU最終評価: trimeshの三角形面最近接で確定計算")
                self._cpu_final_notified = True
        
        elif GPU_AVAILABLE:
            # GPU版：候補生成用（バイアス補正付き）
            rot = R.from_euler("xyz", [rx_rad, ry_rad, 0.0]).as_matrix()
            rot_gpu = array_to_gpu(rot.astype(np.float32))
            
            # ★重要：pivot回りで回転してから平行移動
            # v' = R @ (v - pivot) + pivot + t
            p = self.pivot_gpu
            transformed_gpu = cp.dot((self.v0_gpu - p), rot_gpu.T) + p + cp.array([tx, ty, tz], dtype=cp.float32)
            
            # GPU完全距離計算
            distances_gpu = self._gpu_nearest_distances(transformed_gpu)
            
            # *** 🔧 GPUバイアス補正適用 ***
            distances_corrected = distances_gpu - self.gpu_bias
            distances_corrected = cp.clip(distances_corrected, 0.0, float('inf'))  # 負値クリップ
            
            dist_raw = distances_corrected  # 生距離
            min_dist_raw = float(array_to_cpu(cp.min(dist_raw)))
            
            # ✅ Step0修正: 接触判定は生距離で行う（クリップ前）
            current_threshold = self.contact_threshold_search if self.search_mode else self.contact_threshold_final
            contact_mask_gpu = dist_raw <= current_threshold  # ★判定は生距離
            
            d_gpu = cp.clip(dist_raw, 0.0, max_dist_clip)    # ★クリップは重み計算用

            if not hasattr(self, '_gpu_calc_notified'):
                print(f"🚀 GPU候補生成: 変換（pivot回り） + 距離計算 + バイアス補正({self.gpu_bias:+.3f}mm)")
                if hasattr(cp, 'get_default_memory_pool'):
                    mempool = cp.get_default_memory_pool()
                    print(f"   GPU使用中: {mempool.used_bytes()/(1024*1024):.1f} MB")
                self._gpu_calc_notified = True

        # --------------------------------------------------
        # 1) まったく噛んでいない場合
        #    → 回転・移動ペナルティ + 大きなマイナス定数
        #       （どんな「噛んでいる姿勢」より必ず不利にする）
        # --------------------------------------------------
        # 🔧 修正1: 型混在を防ぐため contact_count は最初から int に統一
        if force_cpu or (not GPU_AVAILABLE):
            contact_count_int = int(np.sum(contact_mask_gpu))
        else:
            contact_count_int = int(array_to_cpu(cp.sum(contact_mask_gpu)))
        
        # 🔥 GPU壊れ検出ガード: 全点接触は物理的に不可能（force_cpu=Trueの場合はスキップ）
        if not force_cpu:  # force_cpu=True時はGPU異常検出をスキップして無限再帰を防ぐ
            if contact_count_int >= len(self.v0) * 0.95:  # 95%以上が接触なら異常
                if not hasattr(self, '_gpu_fallback_notified'):
                    print(f"🔥 GPU異常検出: {contact_count_int}/{len(self.v0)}点が接触扱い → CPU緊急フォールバック")
                    self._gpu_fallback_notified = True
                # CPU再評価で正しい結果を取得
                return self.evaluate(tx, rx_rad, ry_rad, tz, max_dist_clip, force_cpu=True)
        
        if contact_count_int == 0:
            rot_pen = self.rot_penalty * (abs(rx_rad) + abs(ry_rad))
            trans_pen = self.trans_penalty * np.sqrt(tx * tx + tz * tz)

            # 「接触ゼロは最低でも -10 点」くらいにしておく
            score = -(rot_pen + trans_pen) - 10.0

            zero_dict = {name: 0.0 for name in self.region_masks.keys()}
            info = {
                "total_area": 0.0,
                "total_area_eff": 0.0,  # 💡 有効面積追加
                "num_contacts": 0,
                "region_areas": zero_dict,
                "region_scores": zero_dict,
                "left_area": 0.0,
                "right_area": 0.0,
                "anterior_area": 0.0,
                "posterior_area": 0.0,
                "spring_min": 0.0,
                "spring_var": 0.0,
                "spring_mean": 0.0,
                "spring_zero": len(self.valid_regions),
                "tx": tx,
                "rx": rx_rad,
                "ry": ry_rad,
                "tz": tz,
                "min_dist_raw": min_dist_raw,  # 🔍 DEBUG: 診断との整合性確認用
            }
            return score, info

        # --------------------------------------------------
        # 2) ここから下は「接触あり」のケース
        # --------------------------------------------------

        # CPU版とGPU版で適切な配列を使用
        if force_cpu or (not GPU_AVAILABLE):
            # CPU版: numpy配列を使用
            areas_array = self.areas
            contact_mask = contact_mask_gpu  # この時点では既にnumpy array
            d_array = d_gpu  # この時点では既にnumpy array
            
            # contact_mask 部だけの距離・面積
            th = current_threshold
            d_c = d_array[contact_mask]
            w = 1.0 - (d_c / th) ** 2               # d=0 で1, d=th で0
            w = np.clip(w, 0.0, 1.0)

            # 「バネの縮み量 × 断面積」のようなイメージ
            local_strength_c = areas_array[contact_mask] * w

            # 全頂点長の配列に戻す（コンタクト頂点以外は0）
            strength_full = np.zeros_like(areas_array)
            area_full = np.zeros_like(areas_array)
            area_eff_full = np.zeros_like(areas_array)  # 💡 有効面積（重み付き）
            strength_full[contact_mask] = local_strength_c
            area_full[contact_mask] = areas_array[contact_mask]
            area_eff_full[contact_mask] = local_strength_c  # 重み付き面積
            
        else:
            # GPU版: cupy配列を使用
            # contact_mask 部だけの距離・面積
            th = current_threshold
            d_c_gpu = d_gpu[contact_mask_gpu]
            w_gpu = 1.0 - (d_c_gpu / th) ** 2               # d=0 で1, d=th で0
            w_gpu = cp.clip(w_gpu, 0.0, 1.0)

            # 「バネの縮み量 × 断面積」のようなイメージ
            local_strength_c_gpu = self.areas_gpu[contact_mask_gpu] * w_gpu

            # 全頂点長の配列に戻す（コンタクト頂点以外は0）
            strength_full_gpu = cp.zeros_like(self.areas_gpu)
            area_full_gpu = cp.zeros_like(self.areas_gpu)
            area_eff_full_gpu = cp.zeros_like(self.areas_gpu)  # 💡 有効面積（重み付き）
            strength_full_gpu[contact_mask_gpu] = local_strength_c_gpu
            area_full_gpu[contact_mask_gpu] = self.areas_gpu[contact_mask_gpu]
            area_eff_full_gpu[contact_mask_gpu] = local_strength_c_gpu  # 重み付き面積
            
            # GPU版では後でCPUに変換
            strength_full = strength_full_gpu
            area_full = area_full_gpu
            area_eff_full = area_eff_full_gpu

        # ----- バネごとのスコア・面積 -----
        region_scores = {}
        region_areas = {}
        scores_list = []

        # feasible_regions: 接触可能なブロックのみで評価
        feasible_regions = [name for name in self.valid_regions if name not in self.infeasible_regions]
        
        for name in feasible_regions:
            mask = self.region_masks[name]
            
            if force_cpu or (not GPU_AVAILABLE):
                # CPU版: numpy配列で直接計算
                s = float(strength_full[mask].sum())
                a = float(area_full[mask].sum())
            else:
                # GPU版: 事前転送済みGPUマスクを使用（毎回転送しない）
                mask_gpu = self.region_masks_gpu[name]
                s = float(array_to_cpu(cp.sum(strength_full[mask_gpu])))
                a = float(array_to_cpu(cp.sum(area_full[mask_gpu])))
            
            region_scores[name] = s
            region_areas[name] = a
            scores_list.append(s)

        # 頂点が存在しないブロックは 0 扱い（ただしスコア集計には載せない）
        for name in self.region_masks.keys():
            if name not in region_scores:
                region_scores[name] = 0.0
                region_areas[name] = 0.0

        scores_arr = np.array(scores_list, dtype=float)
        total_strength = float(scores_arr.sum())
        
        if force_cpu or (not GPU_AVAILABLE):
            total_area = float(area_full.sum())
            total_area_eff = float(area_eff_full.sum())  # 💡 有効面積
        else:
            total_area = float(array_to_cpu(cp.sum(area_full)))
            total_area_eff = float(array_to_cpu(cp.sum(area_eff_full)))  # 💡 有効面積

        # 5本の輪ゴムの状態（接触不可能ブロックは除外してカウント）
        if len(scores_arr) > 0:
            min_region = float(scores_arr.min())
            var_region = float(scores_arr.var())
            mean_region = float(scores_arr.mean())
            zero_regions = int(np.sum(scores_arr < 1e-6))  # feasibleブロック内での死んだバネ
        else:
            min_region = 0.0
            var_region = 0.0
            mean_region = 0.0
            zero_regions = len(feasible_regions)  # 全feasibleブロックが死亡

        # 左右・前後の合計（ざっくり把握用）
        left_area = region_areas["M_L"] + region_areas["PM_L"]
        right_area = region_areas["M_R"] + region_areas["PM_R"]
        anterior_area = region_areas["ANT"]
        posterior_area = left_area + right_area

        # 回転・移動ペナルティ
        rot_pen = self.rot_penalty * (abs(rx_rad) + abs(ry_rad))
        trans_pen = self.trans_penalty * np.sqrt(tx * tx + tz * tz)

        # ----------------------------
        # 最終スコア
        #   - 全体の噛み込み量（total_strength）
        #   - 最も弱いバネ（min_region）を強く評価
        #   - バネ間のばらつき（var_region）と「死んでいるバネ」の本数を減点
        # ----------------------------
        # 右側窩嵌合ボーナス
        right_bonus = 0.2 * (region_scores.get("M_R", 0) + region_scores.get("PM_R", 0))
        
        score = (
            0.4 * total_strength   # 全体として噛んでいるか（元の成功値）
            + 1.8 * min_region     # 一番弱いバネもちゃんと張っているか（元の成功値）
            - 0.3 * var_region     # 強いバネと弱いバネの差が大きいほど減点（元の成功値）
            - 0.8 * zero_regions   # 完全にサボっているブロックがあると減点（元の成功値）
            + right_bonus          # 右側窩嵌合を促進
            - rot_pen
            - trans_pen
        )

        # contact_count処理（既に int に統一済み）
        info = {
            "total_area": total_area,
            "total_area_eff": total_area_eff,  # 💡 有効面積追加
            "num_contacts": contact_count_int,
            "region_areas": region_areas,
            "region_scores": region_scores,
            "left_area": left_area,
            "right_area": right_area,
            "anterior_area": anterior_area,
            "posterior_area": posterior_area,
            "spring_min": min_region,
            "spring_var": var_region,
            "spring_mean": mean_region,
            "spring_zero": zero_regions,
            "tx": tx,
            "rx": rx_rad,
            "ry": ry_rad,
            "tz": tz,
            "min_dist_raw": min_dist_raw,  # 🔍 DEBUG: 診断との整合性確認用
        }
        return score, info

    def get_contact_points_by_region(self, tx, rx_rad, ry_rad, tz, contact_threshold=None):
        """
        接触点座標とその所属ブロックを取得（可視化用）
        
        Returns
        -------
        contact_points : np.ndarray (N, 3)
            接触点の座標
        region_labels : list of str
            各点の所属ブロック名
        region_summary : dict
            ブロックごとの統計情報
        """
        if contact_threshold is None:
            contact_threshold = self.contact_threshold
        
        ty = 0.0
        rot = R.from_euler("xyz", [rx_rad, ry_rad, 0.0]).as_matrix()
        p = self.pivot
        transformed = (rot @ (self.v0 - p).T).T + p + np.array([tx, ty, tz])
        
        # CPU評価で距離計算
        _, distances, _ = self.upper.nearest.on_surface(transformed)
        dist_raw = np.asarray(distances)
        
        # 接触点のマスク
        contact_mask = (dist_raw <= contact_threshold)
        contact_points = transformed[contact_mask]
        contact_indices = np.where(contact_mask)[0]
        
        # 各点がどのブロックに属するか判定
        region_labels = []
        region_summary = {name: {"count": 0, "min_dist": 999.0, "area": 0.0} 
                         for name in ["M_L", "M_R", "PM_L", "PM_R", "ANT"]}
        
        for idx in contact_indices:
            # 各ブロックのマスクをチェック
            assigned = False
            for name, mask in self.region_masks.items():
                if mask[idx]:
                    region_labels.append(name)
                    region_summary[name]["count"] += 1
                    region_summary[name]["min_dist"] = min(
                        region_summary[name]["min_dist"], 
                        dist_raw[idx]
                    )
                    region_summary[name]["area"] += self.areas[idx]
                    assigned = True
                    break
            if not assigned:
                region_labels.append("UNKNOWN")
        
        return contact_points, region_labels, region_summary


# =============================
# 探索アルゴリズム
# =============================

def update_gpu_bias_dynamic(scorer, tx=0.0, rx=0.0, ry=0.0, tz_samples=None):
    """
    GPU bias補正: 指定された姿勢(tx,rx,ry)でtzをスキャンし、biasを測定
    
    ⚠️  重要: bias測定は初期姿勢(tx=0, rx=0, ry=0)で固定し、全Phaseで再利用することを推奨。
    姿勢ごとに再測定するとbiasが不安定になり、再現性が低下します。
    
    Args:
        scorer: SpringOcclusionScorer インスタンス
        tx, rx, ry: 測定姿勢パラメータ（デフォルトは初期姿勢）
        tz_samples: tzのサンプル点リスト（Noneなら自動生成）
    
    Returns:
        (bias_median, bias_std): 測定されたbiasの中央値と標準偏差
    """
    if not GPU_AVAILABLE:
        return 0.0, 0.0
    
    if tz_samples is None:
        tz_samples = [2.0, 1.0, 0.5, 0.0, -0.5]
    
    bias_list = []
    for tz in tz_samples:
        # CPU診断
        gap_info = scorer.region_gap_info(tx, rx, ry, tz)
        cpu_min = min([info["min"] for info in gap_info.values()])
        
        # GPU評価
        score, info = scorer.evaluate(tx, rx, ry, tz)
        gpu_min = info.get("min_dist_raw", 999.0)
        
        bias = gpu_min - cpu_min
        bias_list.append(bias)
    
    bias_median = np.median(bias_list)
    bias_std = np.std(bias_list)
    
    # scorer のbiasを更新
    scorer.gpu_bias = bias_median
    
    return bias_median, bias_std

def objective_from_info(score, info, scorer, w_lr=1.5, w_pml=0.9, pml_margin=0.10, w_mr=0.3):
    """
    🔧 修正3: 二重評価を防ぐため、score/info から objective を計算
    evaluate() を再度呼ばずに済む
    
    Returns
    -------
    obj : float
        最終的な目的関数値（score + penalties + rewards）
    components : dict
        全penalty/ratio/share成分を含む辞書（デバッグ用）
    """
    rs = info["region_scores"]
    ra = info["region_areas"]
    L = rs["M_L"] + rs["PM_L"]
    R = rs["M_R"] + rs["PM_R"]
    
    # ★ 暴れ抑制1: 接触が少ないときはpen_lrを弱める
    total_strength = L + R
    strength_threshold = 0.05  # この値以下なら接触が少ないと判定
    
    if total_strength < strength_threshold:
        # 接触が少ない → pen_lrを0に近づける（重みを減衰）
        lr_weight_factor = total_strength / strength_threshold  # 0～1
    else:
        lr_weight_factor = 1.0
    
    # ★ 暴れ抑制2: ratio計算にepsを入れて発散防止
    eps = 1e-9
    denom = L + R + eps
    L_ratio = L / denom
    pm_l_share = rs["PM_L"] / (L + eps)
    
    # penalty計算（w_lrは接触量に応じて減衰）
    pen_lr = abs(L_ratio - scorer.target_L_ratio)
    pen_lr_effective = pen_lr * lr_weight_factor  # ★ 減衰適用
    
    excess = max(0.0, pm_l_share - (scorer.target_PM_L_share + pml_margin))
    pen_pml = excess
    mr = rs["M_R"]
    
    # ANT_share（前歯割合）も計算
    total_area = info["total_area"]
    ANT_share = ra["ANT"] / (total_area + eps)
    
    # ★ 前歯過多ペナルティ（40%超で強く罰する）
    ANT_critical = 0.40  # 臨界値：40%超は過多と判定
    ANT_warning = 0.30   # 警告値：30%超で軽く罰する
    if ANT_share > ANT_critical:
        pen_ant = (ANT_share - ANT_critical) * 5.0  # 40%超は強烈に罰する
    elif ANT_share > ANT_warning:
        pen_ant = (ANT_share - ANT_warning) ** 2  # 30-40%は2乗で滑らかに罰する
    else:
        pen_ant = 0.0
    w_ant = 2.0  # 前歯過多ペナルティの重み
    
    # ★ PM_L不足ペナルティ（area基準に変更：面積計算と統一）
    # 接触点数ではなく、PM_L領域の実効面積で判定（整合性向上）
    PM_L_area = ra.get("PM_L", 0.0)
    PM_L_area_min = 0.01  # 目標：最低0.01mm²（約1-2点相当）
    if PM_L_area < PM_L_area_min:
        pen_pml_shortage = (PM_L_area_min - PM_L_area) * 5.0  # 0.01mm²不足につき0.05ペナルティ
    else:
        pen_pml_shortage = 0.0
    w_pml_shortage = 1.0  # PM_L不足ペナルティの重み
    
    # デバッグ用：接触点数もカウント（表示のみ、ペナルティ計算には使わない）
    PM_L_count = sum(1 for label in info.get("contact_labels", []) if label == "PM_L")
    
    # ★ 深噛みガード（めり込み防止）
    min_dist_raw = info.get("min_dist_raw", 999.0)
    pen_deep = 0.0
    deep_bite_warning = False
    
    # 非水密STL検知時は測定誤差を考慮して閾値を緩和
    is_watertight = getattr(scorer, 'mesh_is_watertight', True)  # デフォルトはTrue
    if is_watertight:
        # 水密メッシュ: 厳格な閾値
        critical_threshold = 0.005
        warning_threshold = 0.010
        caution_threshold = 0.015
    else:
        # 非水密メッシュ: 測定誤差を考慮して緩和
        critical_threshold = 0.010  # 5µm → 10µm
        warning_threshold = 0.015   # 10µm → 15µm
        caution_threshold = 0.020   # 15µm → 20µm
    
    # 条件1: 最小距離が危険領域
    if min_dist_raw < critical_threshold:
        pen_deep += (critical_threshold - min_dist_raw) * 100.0  # 強烈なペナルティ
        deep_bite_warning = True
    
    # 条件2: 最小距離が警告領域
    elif min_dist_raw < warning_threshold:
        pen_deep += (warning_threshold - min_dist_raw) * 50.0  # 強めのペナルティ
    
    # 条件3: 最小距離が注意領域
    elif min_dist_raw < caution_threshold:
        pen_deep += (caution_threshold - min_dist_raw) * 10.0  # 軽いペナルティ
    
    w_deep = 1.0  # 深噛みペナルティの重み
    
    # ★ 最終objective（深噛みガード＋ANT過多＋PM_L不足）
    obj = score - w_lr * pen_lr_effective - w_pml * pen_pml + w_mr * mr - w_ant * pen_ant - w_deep * pen_deep - w_pml_shortage * pen_pml_shortage
    
    # 全成分を辞書で返す
    components = {
        "obj": obj,
        "score": score,
        "pen_lr": pen_lr,  # 元の値
        "pen_lr_effective": pen_lr_effective,  # 減衰後
        "lr_weight_factor": lr_weight_factor,  # 減衰係数
        "pen_pml": pen_pml,
        "pen_pml_shortage": pen_pml_shortage,  # ★ PM_L不足ペナルティ
        "PM_L_area": PM_L_area,  # ★ PM_L面積（判定基準）
        "PM_L_count": PM_L_count,  # ★ PM_L接触点数（参考値）
        "pen_ant": pen_ant,  # ★ 前歯過多ペナルティ
        "pen_deep": pen_deep,  # ★ 深噛みペナルティ
        "min_dist_raw": min_dist_raw,  # ★ 最小距離（診断用）
        "deep_bite_warning": deep_bite_warning,  # ★ 深噛み警告フラグ
        "excess": excess,
        "mr": mr,
        "L_ratio": L_ratio,
        "pm_l_share": pm_l_share,
        "ANT_share": ANT_share,
        "ANT_critical": ANT_share > ANT_critical,  # ★ 前歯過多フラグ（40%超）
        "dead": info["spring_zero"],
        "total_strength": total_strength,
    }
    
    return obj, components

def line_search_tz(scorer: SpringOcclusionScorer,
                   tx0=0.0, rx0=0.0, ry0=0.0,
                   tz_start=0.5, tz_end=-1.5, step=-0.05,
                   # ★バランス補正の重み（M_R優勢を抑制する設定）
                   w_lr=1.5,          # 左右バランス（1.2→1.5に増強）
                   w_pml=0.9,         # 左小臼歯（PM_L）の偏り抑制（0.8→0.9）
                   pml_margin=0.10,   # "許容する"PM_L share の余裕
                   w_mr=0.3           # 右大臼歯（M_R）報酬を減らす（0.4→0.3）
                   ):
    """
    tz 方向にまっすぐ閉口しながら、
    score最大ではなく「score + バランス補正」を最大化する tz を探す
    → これをヒルクライムの初期値にする
    """

    def objective(tx, rx, ry, tz):
        score, info = scorer.evaluate(tx, rx, ry, tz)
        # 🔧 修正3: objective_from_info を使って二重評価を回避
        obj, comp = objective_from_info(score, info, scorer, w_lr, w_pml, pml_margin, w_mr)
        return obj, score, info, comp["L_ratio"], comp["pm_l_share"]

    best_obj = -1e18
    best_score = -1e18
    best_tz = tz_start
    best_info = None

    tz = tz_start
    print("\n[Step1] tz 方向スキャンで初期位置を探索（objective で選択）")
    i = 0
    while tz >= tz_end - 1e-9:
        obj, score, info, L_ratio, pm_l_share = objective(tx0, rx0, ry0, tz)

        if i % 5 == 0:
            ra = info["region_areas"]
            rs = info["region_scores"]
            min_raw = info.get("min_dist_raw", 999.0)  # 🔍 DEBUG: GPU vs 診断の整合性確認
            area_eff = info.get("total_area_eff", 0.0)  # 💡 有効面積表示
            print(
                f"  tz={tz:6.3f} mm -> obj={obj:7.3f}, score={score:7.3f}, "
                f"area={info['total_area']:.4f}, area_eff={area_eff:.4f}, min_dist={min_raw:.4f}mm | "
                f"L_ratio={L_ratio:.3f}, PM_L_share={pm_l_share:.3f} | "
                f"[str] M_R={rs['M_R']:.4f}, PM_L={rs['PM_L']:.4f} | "
                f"[area] M_L={ra['M_L']:.3f}, M_R={ra['M_R']:.3f}, "
                f"PM_L={ra['PM_L']:.3f}, PM_R={ra['PM_R']:.3f}, ANT={ra['ANT']:.3f}"
            )

        if obj > best_obj:
            best_obj = obj
            best_score = score
            best_tz = tz
            best_info = info

        tz += step
        i += 1

    print(
        f"\n  → 初期候補: tz={best_tz:.3f} mm, obj={best_obj:.3f}, score={best_score:.3f}, "
        f"area={best_info['total_area']:.4f}"
    )
    
    # 🎯 方式A': Step1最終候補をCPUで確定評価
    print(f"🎯 Step1最終候補をCPU確定評価: tz={best_tz:.3f}mm")
    cpu_score, cpu_info = scorer.evaluate(tx0, rx0, ry0, best_tz, force_cpu=True)
    print(f"   GPU候補: score={best_score:.3f}, area={best_info['total_area']:.4f}mm², contacts={best_info['num_contacts']}")
    print(f"   CPU確定: score={cpu_score:.3f}, area={cpu_info['total_area']:.4f}mm², contacts={cpu_info['num_contacts']}")
    
    return best_tz, cpu_score, cpu_info


def hill_climb_4d(scorer: SpringOcclusionScorer,
                  tx_init, rx_init, ry_init, tz_init,
                  tx_step=0.05, deg_step=0.5, tz_step=0.05,
                  max_iter=20,
                  tx_min=-0.8, tx_max=0.8,
                  max_rot_deg=5.0,
                  tz_min=-2.0, tz_max=1.0,
                  # ★バランス補正の重み（M_R優勢を抑制する設定）
                  w_lr=1.5,          # 左右バランス（1.2→1.5に増強）
                  w_pml=0.9,         # 左小臼歯（PM_L）偏り抑制（0.8→0.9）
                  pml_margin=0.10,   # PM_L share "許容マージン"
                  w_mr=0.3,          # 右大臼歯（M_R）報酬を減らす（0.4→0.3）
                  force_cpu_eval=False  # ★CPU確定評価モード
                  ):
    """
    (tx, rx, ry, tz) の4自由度ヒルクライム
    ただし比較は score ではなく objective（score + バランス補正）で行う
    force_cpu_eval=True のとき、全評価をCPUで行う（確定モード用）
    """

    # ✅ 評価モードを決定：force_cpu_eval=True → mode="strict"
    eval_mode = "strict" if force_cpu_eval else "search"

    tx = tx_init
    rx = rx_init
    ry = ry_init
    tz = tz_init

    # ✅ 外部objective()関数を使用（mode統一）
    obj, score, info = objective(tx, rx, ry, tz, scorer, w_lr, w_pml, pml_margin, w_mr, mode=eval_mode)
    _, comp = objective_from_info(score, info, scorer, w_lr, w_pml, pml_margin, w_mr)
    L_ratio = comp["L_ratio"]
    pm_l_share = comp["pm_l_share"]
    
    print("\n[Step2] 近傍ヒルクライム開始（objective で最適化）")
    print(
        f"  start: tx={tx:.3f}mm, rx={np.rad2deg(rx):.3f}°, "
        f"ry={np.rad2deg(ry):.3f}°, tz={tz:.3f} mm, "
        f"obj={obj:.3f}, score={score:.3f}, area={info['total_area']:.4f}, "
        f"L_ratio={L_ratio:.3f}, PM_L_share={pm_l_share:.3f}"
    )
    
    # デバッグ用：初期評価の詳細表示
    if force_cpu_eval:
        print(f"  [DEBUG] 初期 obj={obj:.6f} (この値より大きい obj を探す)")

    rad_step = np.deg2rad(deg_step)
    max_rot_rad = np.deg2rad(max_rot_deg)

    for it in range(max_iter):
        improved = False
        best_local_obj = obj
        best_local = (tx, rx, ry, tz)
        best_local_score = score
        best_local_info = info
        best_lr = L_ratio
        best_pml = pm_l_share

        # GPU最適化：近傍候補をまとめて評価
        neighbor_poses = []
        for d_tx in [-tx_step, 0.0, tx_step]:
            for d_rx in [-rad_step, 0.0, rad_step]:
                for d_ry in [-rad_step, 0.0, rad_step]:
                    for d_tz in [-tz_step, 0.0, tz_step]:
                        if d_tx == 0.0 and d_rx == 0.0 and d_ry == 0.0 and d_tz == 0.0:
                            continue

                        tx_c = tx + d_tx
                        rx_c = rx + d_rx
                        ry_c = ry + d_ry
                        tz_c = tz + d_tz

                        # 範囲制限
                        if tx_c < tx_min or tx_c > tx_max:
                            continue
                        if abs(rx_c) > max_rot_rad or abs(ry_c) > max_rot_rad:
                            continue
                        if tz_c < tz_min or tz_c > tz_max:
                            continue

                        neighbor_poses.append([tx_c, rx_c, ry_c, tz_c])
        
        # ★ 追加：tz を同時に動かす近傍も試す（停滞対策）
        if len(neighbor_poses) > 0:
            # ±tz_step だけ動かした近傍をいくつか追加
            for i, pose in enumerate(neighbor_poses[:min(5, len(neighbor_poses))]):  # 最初の5候補だけ
                tx_c, rx_c, ry_c, tz_c = pose
                for d_tz_extra in [-tz_step, tz_step]:
                    tz_extra = tz_c + d_tz_extra
                    if tz_min <= tz_extra <= tz_max:
                        neighbor_poses.append([tx_c, rx_c, ry_c, tz_extra])
        
        if neighbor_poses:
            # バッチ評価でGPU加速
            neighbor_poses = np.array(neighbor_poses)
            # ★ 確定モードCPUではバッチGPU評価を使わない
            if (not force_cpu_eval) and GPU_AVAILABLE and len(neighbor_poses) > 2:  # より積極的にGPUバッチ評価を使用
                batch_scores, batch_infos = scorer.evaluate_batch(neighbor_poses)
                
                for i, (pose, score_c, info_c) in enumerate(zip(neighbor_poses, batch_scores, batch_infos)):
                    tx_c, rx_c, ry_c, tz_c = pose
                    # 🔧 修正3: objective_from_info を使って二重評価を回避
                    obj_c, comp_c = objective_from_info(score_c, info_c, scorer, w_lr, w_pml, pml_margin, w_mr)
                    
                    if obj_c > best_local_obj:
                        best_local_obj = obj_c
                        best_local = (tx_c, rx_c, ry_c, tz_c)
                        best_local_score = score_c
                        best_local_info = info_c
                        best_lr = comp_c["L_ratio"]
                        best_pml = comp_c["pm_l_share"]
                        improved = True
            else:
                # CPU fallback または少数候補の場合（確定モードCPUもここ）
                for pose in neighbor_poses:
                    tx_c, rx_c, ry_c, tz_c = pose
                    score_c, info_c = scorer.evaluate(tx_c, rx_c, ry_c, tz_c, force_cpu=force_cpu_eval)
                    # 🔧 修正3: objective_from_info を使って二重評価を回避
                    obj_c, comp_c = objective_from_info(score_c, info_c, scorer, w_lr, w_pml, pml_margin, w_mr)

                    # デバッグ用：近傍評価の詳細表示（CPU評価時のみ）
                    if force_cpu_eval and obj_c > best_local_obj:
                        print(f"    [DEBUG] 改善候補 pose=({tx_c:.3f}, {np.rad2deg(rx_c):.2f}°, {np.rad2deg(ry_c):.2f}°, {tz_c:.3f}), obj={obj_c:.6f} (vs {best_local_obj:.6f}, 差分={obj_c - best_local_obj:.6f})")

                    if obj_c > best_local_obj:
                        best_local_obj = obj_c
                        best_local = (tx_c, rx_c, ry_c, tz_c)
                        best_local_score = score_c
                        best_local_info = info_c
                        best_lr = comp_c["L_ratio"]
                        best_pml = comp_c["pm_l_share"]
                        improved = True

        if not improved:
            # ★ 最低反復回数保証（it<2では継続探索、処理時間とのバランス）
            if it < 2:
                print(f"  it={it}: 改善なし → 継続探索（最低反復2回未達、刻み幅を縮小）")
                # 刻み幅を少し縮小して継続
                tx_step *= 0.75
                rad_step *= 0.75
                tz_step *= 0.75
                continue
            else:
                print(f"  it={it}: 改善なし → 終了")
                break

        tx, rx, ry, tz = best_local
        obj = best_local_obj
        score = best_local_score
        info = best_local_info
        L_ratio = best_lr
        pm_l_share = best_pml

        ra = info["region_areas"]
        rs = info["region_scores"]
        print(
            f"  it={it+1}: tx={tx:6.3f}mm, rx={np.rad2deg(rx):5.2f}°, "
            f"ry={np.rad2deg(ry):5.2f}°, tz={tz:6.3f} mm, "
            f"obj={obj:7.3f}, score={score:7.3f}, area={info['total_area']:.4f}, "
            f"L_ratio={L_ratio:.3f}, PM_L_share={pm_l_share:.3f}, "
            f"[str] M_R={rs['M_R']:.4f}, PM_L={rs['PM_L']:.4f}"
        )

    # ★ 修正：細かい刻みで最終リファイン（GPU評価→上位のみCPU確定）
    if force_cpu_eval and it < max_iter - 1:
        print(f"\n  🔬 細かい刻みで最終リファイン（刻み: tx={tx_step/2:.3f}, deg={deg_step/2:.2f}°, tz={tz_step/2:.3f}）")
        fine_tx_step = tx_step / 2
        fine_rad_step = rad_step / 2
        fine_tz_step = tz_step / 2
        
        for fine_it in range(2):  # 最大2回（高速化）
            # 1) GPU評価で全候補を高速スクリーニング
            candidates = []
            for d_tx in [-fine_tx_step, 0.0, fine_tx_step]:
                for d_rx in [-fine_rad_step, 0.0, fine_rad_step]:
                    for d_ry in [-fine_rad_step, 0.0, fine_rad_step]:
                        for d_tz in [-fine_tz_step, 0.0, fine_tz_step]:
                            if d_tx == 0.0 and d_rx == 0.0 and d_ry == 0.0 and d_tz == 0.0:
                                continue
                            
                            tx_c = tx + d_tx
                            rx_c = rx + d_rx
                            ry_c = ry + d_ry
                            tz_c = tz + d_tz
                            
                            if tx_c < tx_min or tx_c > tx_max:
                                continue
                            if abs(rx_c) > max_rot_rad or abs(ry_c) > max_rot_rad:
                                continue
                            if tz_c < tz_min or tz_c > tz_max:
                                continue
                            
                            # GPU評価（高速近似）
                            score_g, info_g = scorer.evaluate(tx_c, rx_c, ry_c, tz_c, force_cpu=False)
                            obj_g, _ = objective_from_info(score_g, info_g, scorer, w_lr, w_pml, pml_margin, w_mr)
                            candidates.append((obj_g, tx_c, rx_c, ry_c, tz_c))
            
            if not candidates:
                print(f"    fine_it={fine_it}: 候補なし → 終了")
                break
            
            # 2) 上位TOP_K個だけCPU確定（厳密評価）
            candidates.sort(reverse=True, key=lambda x: x[0])
            TOP_K = 8  # 5〜10推奨
            print(f"    fine_it={fine_it}: GPU評価で{len(candidates)}候補 → 上位{min(TOP_K, len(candidates))}個をCPU確定中...")
            
            improved_fine = False
            best_local_obj_fine = obj
            best_local_fine = (tx, rx, ry, tz)
            
            for obj_g, tx_c, rx_c, ry_c, tz_c in candidates[:TOP_K]:
                score_c, info_c = scorer.evaluate(tx_c, rx_c, ry_c, tz_c, force_cpu=True)
                obj_c, comp_c = objective_from_info(score_c, info_c, scorer, w_lr, w_pml, pml_margin, w_mr)
                
                if obj_c > best_local_obj_fine:
                    best_local_obj_fine = obj_c
                    best_local_fine = (tx_c, rx_c, ry_c, tz_c)
                    score = score_c
                    info = info_c
                    L_ratio = comp_c["L_ratio"]
                    pm_l_share = comp_c["pm_l_share"]
                    improved_fine = True
            
            if improved_fine:
                tx, rx, ry, tz = best_local_fine
                obj = best_local_obj_fine
                print(f"    fine_it={fine_it}: obj={obj:.6f} (改善)")
            else:
                print(f"    fine_it={fine_it}: 改善なし → 細かい探索終了")
                break

    # 🎯 最終候補をCPUで確定評価（GPUバイアス問題を回避）
    if scorer.use_cpu_final_eval and GPU_AVAILABLE:
        print(f"\n🎯 最終候補 (tx={tx:.3f}, tz={tz:.3f}) をCPUで確定評価中...")
        final_score, final_info = scorer.evaluate(tx, rx, ry, tz, force_cpu=True)
        print(f"   GPU評価: score={score:.3f}, area={info['total_area']:.4f}mm²")
        print(f"   CPU確定: score={final_score:.3f}, area={final_info['total_area']:.4f}mm²")
        return tx, rx, ry, tz, final_score, final_info
    
    # 返す score/info は "純 score" のもの（従来互換）
    return tx, rx, ry, tz, score, info


# =============================
# メイン
# =============================

def main():
    import time
    start_time = time.perf_counter()  # 処理開始時刻
    
    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(
        description="咬頭嵌合位自動最適化（5本の輪ゴムスプリングモデル）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python app_gyu.py                  # ダイアログで選択
  python app_gyu.py --move lower     # 明示的に下顎を動かす（上顎固定）
  python app_gyu.py --move upper     # 明示的に上顎を動かす（下顎固定）
        """
    )
    parser.add_argument(
        "--move",
        choices=["lower", "upper"],
        default=None,  # Noneにして、指定がなければダイアログで選択
        help="動かす顎を選択 (指定なしの場合はダイアログで選択)"
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print("咬頭嵌合位自動最適化（5本の輪ゴムスプリングモデル）v4 - 診断強化版")
    print("=" * 80)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 出力形式を決定（コマンドライン引数またはダイアログ）
    # ★重要：最適化は常に「下顎移動モード」で実行（安定性・再現性が高い）
    #        output_mode は「どちらのSTLを動かして出力するか」の選択
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if args.move is None:
        # コマンドライン引数がない場合はダイアログで選択
        output_mode = select_moving_jaw()
    else:
        output_mode = args.move
        print(f"🦷 出力モード: {output_mode}（{'下顎' if output_mode == 'lower' else '上顎'}）を動かした結果を出力")

    print(f"\n📌 最適化方式: 常に「下顎移動」で実行（安定性・再現性が最も高い）")
    print(f"📌 出力形式: {'下顎を動かす（A適用）' if output_mode == 'lower' else '上顎を動かす（A⁻¹適用）'}")
    print(f"   → 相対咬合は完全に同一、座標系だけ違う")

    upper_path, lower_path = select_two_stl_files()
    upper = load_mesh_safely(upper_path)
    lower = load_mesh_safely(lower_path)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 最適化は常に「下顎移動モード」で実行（固定）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n頂点面積を計算中...")
    sample_mesh = lower  # 常に下顎をサンプル（動かす側）
    fixed_mesh = upper   # 常に上顎を固定
    print("📌 最適化用メッシュ設定: 下顎サンプル（移動）/ 上顎固定")
    
    sample_vertex_area_all = per_vertex_area(sample_mesh)
    all_vertices = sample_mesh.vertices
    n_vertices = len(all_vertices)
    SAMPLE_SIZE = 1200  # 元の成功値（精度と速度のバランス）

    if n_vertices > SAMPLE_SIZE:
        rng = np.random.default_rng(0)
        sample_idx = rng.choice(n_vertices, size=SAMPLE_SIZE, replace=False)
        print(f"✓ {n_vertices} 頂点から {SAMPLE_SIZE} 頂点をサンプリング")
    else:
        sample_idx = np.arange(n_vertices)
        print(f"✓ 全 {n_vertices} 頂点を使用（{n_vertices} 頂点）")

    sample_vertices = all_vertices[sample_idx]
    sample_areas = sample_vertex_area_all[sample_idx]

    # 回転中心：下顎メッシュ全体の重心（evaluate()とSTL出力で一致させる）
    pivot_lower = lower.vertices.mean(axis=0)
    print(f"🎯 回転中心（pivot）設定: [{pivot_lower[0]:.3f}, {pivot_lower[1]:.3f}, {pivot_lower[2]:.3f}]")

    # スコアラー準備
    # 常に「上顎固定、下顎移動」で最適化
    scorer = SpringOcclusionScorer(
        upper_mesh=fixed_mesh,  # 上顎（固定側）
        lower_sample_vertices=sample_vertices,  # 下顎サンプル（動かす側）
        lower_sample_areas=sample_areas,
        contact_threshold=0.035,  # 右側窩嵌合改善のため微増（0.035mm）
        rot_penalty=1.5,
        trans_penalty=2.0,
        moving_jaw="lower",  # 最適化は常に下顎移動
        lower_mesh_for_springs=None,  # 下顎移動なのでNone（サンプル頂点から定義）
        pivot=pivot_lower,  # ★重要：evaluate()と出力で同じ変換を使う
    )

    # まず、メッシュの大まかな位置関係を調査
    print("\n[診断] メッシュ位置関係の調査")
    upper_center = upper.vertices.mean(axis=0)
    lower_center = lower.vertices.mean(axis=0)
    print(f"上顎中心: {upper_center}")
    print(f"下顎中心: {lower_center}")
    print(f"初期ギャップ: {upper_center - lower_center}")
    
    # より広い範囲での距離調査（Step1範囲を自動決定するため）
    test_positions = [
        (0.0, 0.0, 0.0, 2.0),   # 大きく離れた位置
        (0.0, 0.0, 0.0, 1.0),   # 中間位置  
        (0.0, 0.0, 0.0, 0.0),   # 基準位置
        (0.0, 0.0, 0.0, -1.0),  # 接近位置
        (0.0, 0.0, 0.0, -2.0),  # さらに接近
    ]
    
    print("\n[診断] 各位置での最短距離調査 (CPUの診断):")
    tz_distance_map = {}
    for tx, rx, ry, tz in test_positions:
        gap_info = scorer.region_gap_info(tx, rx, ry, tz, near_th=0.5)  # 0.5mm以内
        min_distances = [info["min"] for info in gap_info.values()]
        overall_min = min(min_distances)
        tz_distance_map[tz] = overall_min
        print(f"  tz={tz:5.1f}mm: 最短距離={overall_min:.3f}mm")
        if overall_min < 0.1:  # 0.1mm以内なら詳細表示
            for region, info in gap_info.items():
                if info["min"] < 0.1:
                    print(f"    {region}: min={info['min']:.3f}mm, near_count={info['near_count']}")
    
    # ★ Step1のtzスキャン範囲を診断結果から自動決定
    # 接触域（min_dist < 0.2mm）を含むようにする
    tz_values = sorted(tz_distance_map.keys())
    contact_tzs = [tz for tz, dist in tz_distance_map.items() if dist < 0.2]
    
    if contact_tzs:
        tz_contact_min = min(contact_tzs)
        tz_contact_max = max(contact_tzs)
        # 接触域を中心に±0.5mm余裕を持たせる
        tz_start_auto = tz_contact_max + 0.5
        tz_end_auto = tz_contact_min - 0.5
        print(f"\n✓ [自動設定] 接触域検出: tz={tz_contact_min:.1f}~{tz_contact_max:.1f}mm")
        print(f"  → Step1スキャン範囲: tz={tz_start_auto:.1f} → {tz_end_auto:.1f}mm")
    else:
        # 接触域が見つからない場合はデフォルト
        tz_start_auto = 1.5
        tz_end_auto = -1.0
        print(f"\n⚠️  接触域未検出 → デフォルト範囲: tz={tz_start_auto:.1f} → {tz_end_auto:.1f}mm")
    
    # 閉口方向も診断から自動判定
    # tzを変化させたときの距離変化から判定
    tz_sorted = sorted(tz_distance_map.items())
    if len(tz_sorted) >= 2:
        # 距離が最も減る方向を閉口方向とする
        min_tz = min(tz_sorted, key=lambda x: x[1])[0]
        if min_tz < 0:
            closing_direction = "tz-"
            closing_sign = -1
        else:
            closing_direction = "tz+"
            closing_sign = +1
        print(f"✓ [自動判定] 閉口方向: {closing_direction} (最短距離のtz={min_tz:.1f}mm)")
    else:
        closing_direction = "tz-"
        closing_sign = -1
        print(f"⚠️  閉口方向判定不能 → デフォルト: {closing_direction}")

    print("\n🔍 [重要診断] GPU vs CPU の距離計算整合性テスト:")
    print("  ⚠️  重要: 以下の測定は tx=0, rx=0, ry=0 の固定条件で実施")
    print("         全Phaseでこの値を再利用し、姿勢ごとの再測定は行いません")
    print("         （理由: 姿勢ごとの再測定はbiasを不安定化させ、再現性を低下させるため）")
    print("="*80)
    print("検証: 接触域でGPUバイアスの特性を確認")
    bias_list = []
    
    # 接触域の代表値でテスト
    if contact_tzs:
        test_tz_values = [tz_contact_max, (tz_contact_max + tz_contact_min)/2, tz_contact_min]
    else:
        test_tz_values = [1.0, 0.5, 0.0, -0.5, -1.0]
    
    for tz in test_tz_values:
        tx, rx, ry = 0.0, 0.0, 0.0
        # CPU診断
        gap_info = scorer.region_gap_info(tx, rx, ry, tz)
        cpu_min = min([info["min"] for info in gap_info.values()])
        
        # GPU評価
        score, info = scorer.evaluate(tx, rx, ry, tz)
        gpu_min = info.get("min_dist_raw", 999.0)
        
        bias = gpu_min - cpu_min
        bias_list.append(bias)
        
        print(f"  tz={tz:5.2f}mm: CPU={cpu_min:.4f}mm, GPU={gpu_min:.4f}mm, バイアス={bias:+.4f}mm")
        if abs(bias) > 0.01:  # 0.01mm以上の差があれば警告
            if abs(bias) > 0.03:  # 閾値以上なら接触判定に影響
                print(f"    🔥 CRITICAL: バイアス{bias:+.4f}mmが閾値0.035mmを超過！接触判定が破綻")
            else:
                print(f"    ⚠️  バイアス{bias:+.4f}mmが検出（接触判定に影響する可能性）")
    
    # バイアス分析
    bias_arr = np.array(bias_list)
    bias_median = np.median(bias_arr)
    bias_std = np.std(bias_arr)
    bias_range = np.max(bias_arr) - np.min(bias_arr)
    
    print(f"\n📊 バイアス分析結果:")
    print(f"  中央値: {bias_median:+.4f}mm")
    print(f"  標準偏差: {bias_std:.4f}mm")
    print(f"  範囲: {bias_range:.4f}mm")
    
    if bias_std < 0.003:  # より厳しい基準
        print(f"  ✓ 極めて安定（bias補正方式C推奨）: GPU距離から {bias_median:.4f}mm を引けば修正")
        scorer.gpu_bias = bias_median  
        correction_method = "C"
    elif bias_std < 0.008:
        print(f"  ○ ある程度安定（bias補正＋CPU確定方式A推奨）")
        scorer.gpu_bias = bias_median  
        correction_method = "AC"
    else:
        print(f"  ⚠️  不安定（CPU確定方式A＋探索緩和方式B推奨）")
        scorer.gpu_bias = bias_median  
        correction_method = "AB"
    
    print(f"\n🔧 採用対策: 方式A（GPU候補生成＋CPU最終確定）＋方式B（探索時閾値調整）＋方式C（GPUバイアス補正）")
    print(f"   探索時: contact_threshold = 0.040mm（バイアス補正により締められる）")
    print(f"   確定時: contact_threshold = 0.035mm（精度重視、CPU評価）")
    print(f"   GPUバイアス: +{scorer.gpu_bias:.4f}mm → CPU相当の距離感に補正")
    
    # 対策B+C: 探索時の閾値（バイアス補正が安定しているため0.050→0.040に締められる）
    scorer.contact_threshold_search = 0.040  # バイアス補正により締めても安定
    scorer.contact_threshold_final = scorer.contact_threshold  # 元の厳密閾値を保存

    # � 接触可能性診断: 絶対当たらない歯を探索から除外  
    scorer.update_feasibility(tx_range=(-0.8, 0.8), tz_range=(-2.0, 2.0))
    
    print(f"\n🔧 探索モード開始: contact_threshold = {scorer.contact_threshold_search:.3f}mm（安定性重視）")
    scorer.search_mode = True  # 探索モード有効化
    
    # ⚠️  biasは初期診断時に固定されているため、ここでは再測定せず全Phaseで再利用
    print(f"\n🔧 Phase1開始: GPU bias={scorer.gpu_bias:+.4f}mm（初期診断から再利用）")
    
    # Step1: tz 方向スキャンで初期位置（診断結果から自動決定した範囲を使用）
    best_tz, best_score_tz, info_tz = line_search_tz(
        scorer,
        tx0=0.0,
        rx0=0.0,
        ry0=0.0,
        tz_start=tz_start_auto,  # 診断結果から自動決定
        tz_end=tz_end_auto,      # 診断結果から自動決定
        step=-0.05
    )

    # Step2 (Phase1): マルチスタート近傍ヒルクライム（局所最適からの脱出）
    print(f"\n{'='*80}")
    print("[Phase1] マルチスタート近傍ヒルクライム（txも含めて最適化）")
    print(f"{'='*80}")
    print(f"  初期位置パターン: [中央, 左寄り, 右寄り] から探索し、最良を選択")
    
    # 3つの初期位置パターン
    start_patterns = [
        {"name": "中央", "tx": 0.0},
        {"name": "左寄り", "tx": -0.2},
        {"name": "右寄り", "tx": +0.2},
    ]
    
    best_overall = None
    best_overall_score = -999.0
    
    for pattern in start_patterns:
        print(f"\n  ★ 初期位置パターン: {pattern['name']} (tx={pattern['tx']:.2f})")
        
        tx_c, rx_c, ry_c, tz_c, score_c, info_c = hill_climb_4d(
            scorer,
            tx_init=pattern["tx"],
            rx_init=0.0,
            ry_init=0.0,
            tz_init=best_tz,
            tx_step=0.05,
            deg_step=0.5,
            tz_step=0.05,
            max_iter=20,
            tx_min=-0.8,
            tx_max=0.8,
            max_rot_deg=5.0,
            tz_min=-2.0,
            tz_max=2.0,
        )
        
        print(f"    → 結果: tx={tx_c:.3f}, tz={tz_c:.3f}, score={score_c:.3f}")
        
        if score_c > best_overall_score:
            best_overall_score = score_c
            best_overall = (tx_c, rx_c, ry_c, tz_c, score_c, info_c)
            print(f"    ★ 最良更新! (score={score_c:.3f})")
    
    # 最良のパターンを採用
    tx_best, rx_best, ry_best, tz_best, score_best, info_best = best_overall
    print(f"\n  ★ Phase1マルチスタート最良: tx={tx_best:.3f}, tz={tz_best:.3f}, score={score_best:.3f}")
    
    # 🎯 対策A: 探索完了後、確定モードでCPU最終評価
    print(f"\n🎯 確定モード切替: contact_threshold = {scorer.contact_threshold_final:.3f}mm（精度重視）")
    scorer.search_mode = False  # 確定モード（厳密閾値）に切り替え

    # ✅ Phase1結果を CPU strict（Phase2/Phase3と同一関数）で再評価
    print(f"\n🔍 Phase1最終候補をCPU strict (0.035mm) で再評価...")
    score_best, info_best = scorer.evaluate(tx_best, rx_best, ry_best, tz_best, force_cpu=True)
    
    # 🔍 検査ログ: Phase1最終姿勢と評価を記録
    print(f"\n[POSE phase1_final] tx={tx_best:.3f} rx={np.rad2deg(rx_best):.3f}° ry={np.rad2deg(ry_best):.3f}° tz={tz_best:.3f}")
    print(f"[STRICT phase1_final] score={score_best:.3f} area={info_best['total_area']:.4f} contacts={info_best['num_contacts']} dead={info_best['spring_zero']}")
    ra = info_best["region_areas"]
    print(f"  area_by_region: M_L={ra['M_L']:.4f} M_R={ra['M_R']:.4f} PM_L={ra['PM_L']:.4f} PM_R={ra['PM_R']:.4f} ANT={ra['ANT']:.4f}")

    print("\nPhase1 結果（ノーマル咬合位置）")
    print("-" * 80)
    print(f"  tx = {tx_best:6.3f} mm")
    print(f"  rx = {np.rad2deg(rx_best):6.3f} °")
    print(f"  ry = {np.rad2deg(ry_best):6.3f} °")
    print(f"  tz = {tz_best:6.3f} mm")
    print(f"  score           = {score_best:.3f}")
    print(f"  total area      = {info_best['total_area']:.4f} mm²")
    print(f"  M_L area        = {ra['M_L']:.4f} mm²")
    print(f"  M_R area        = {ra['M_R']:.4f} mm²")
    print(f"  PM_L area       = {ra['PM_L']:.4f} mm²")
    print(f"  PM_R area       = {ra['PM_R']:.4f} mm²")
    print(f"  ANT area        = {ra['ANT']:.4f} mm²")
    print(f"  contacts        = {info_best['num_contacts']} points")
    print(f"  spring min      = {info_best['spring_min']:.4f}")
    print(f"  spring var      = {info_best['spring_var']:.4f}")
    print(f"  dead springs    = {info_best['spring_zero']}")
    print(f"  🔍 min_dist_raw = {info_best.get('min_dist_raw', 'N/A'):.4f} mm")
    print("-" * 80)

    rs = info_best["region_scores"]
    print("\n  [region_scores (strength)]")
    print(f"  M_L={rs['M_L']:.6f}, M_R={rs['M_R']:.6f}, "
          f"PM_L={rs['PM_L']:.6f}, PM_R={rs['PM_R']:.6f}, ANT={rs['ANT']:.6f}")

    left_s  = rs["M_L"] + rs["PM_L"]
    right_s = rs["M_R"] + rs["PM_R"]
    denom = left_s + right_s + 1e-9
    print(f"  L_strength={left_s:.6f}, R_strength={right_s:.6f}, "
          f"L_ratio={left_s/denom:.3f}")

    # ★ Phase2: tz だけを少し「ギュッ」と噛み込ませる（CPU確定モードで実行）
    print(f"\n🔧 Phase2 CPU確定モード: contact_threshold = 0.035mm（Phase3と同じ評価関数）")
    scorer.search_mode = False  # ✅ CPU確定モードに切り替え（Phase3と同一条件）
    
    tz_gyu, score_gyu, info_gyu = gyu_refine_tz(
        scorer,
        tx_best, rx_best, ry_best, tz_best,
        extra_depth=0.10,  # ← ギュッとする最大量（mm）。0.05〜0.10 あたりから調整
        step=-0.01,        # 0.01mm 刻み
        closing_sign=closing_sign,  # 診断から自動判定された閉口方向
    )

    print("\n最終結果（Phase2: ちょっとギュッ後）")
    print("-" * 80)
    print(f"  tx = {tx_best:6.3f} mm")              # tx, rx, ry は Phase1 のまま
    print(f"  rx = {np.rad2deg(rx_best):6.3f} °")
    print(f"  ry = {np.rad2deg(ry_best):6.3f} °")
    print(f"  tz = {tz_gyu:6.3f} mm")              # tz だけ gyu 版
    print(f"  score           = {score_gyu:.3f}")
    print(f"  total area      = {info_gyu['total_area']:.4f} mm²")
    ra2 = info_gyu["region_areas"]
    print(f"  M_L area        = {ra2['M_L']:.4f} mm²")
    print(f"  M_R area        = {ra2['M_R']:.4f} mm²")
    print(f"  PM_L area       = {ra2['PM_L']:.4f} mm²")
    print(f"  PM_R area       = {ra2['PM_R']:.4f} mm²")
    print(f"  ANT area        = {ra2['ANT']:.4f} mm²")
    print(f"  contacts        = {info_gyu['num_contacts']} points")
    print(f"  spring min      = {info_gyu['spring_min']:.4f}")
    print(f"  spring var      = {info_gyu['spring_var']:.4f}")
    print(f"  dead springs    = {info_gyu['spring_zero']}")
    print(f"  🔍 min_dist_raw = {info_gyu.get('min_dist_raw', 'N/A'):.4f} mm")
    print("-" * 80)

    rs2 = info_gyu["region_scores"]
    print("\n  [region_scores (strength)]")
    print(f"  M_L={rs2['M_L']:.6f}, M_R={rs2['M_R']:.6f}, "
          f"PM_L={rs2['PM_L']:.6f}, PM_R={rs2['PM_R']:.6f}, ANT={rs2['ANT']:.6f}")

    left_s2  = rs2["M_L"] + rs2["PM_L"]
    right_s2 = rs2["M_R"] + rs2["PM_R"]
    denom2 = left_s2 + right_s2 + 1e-9
    print(f"  L_strength={left_s2:.6f}, R_strength={right_s2:.6f}, "
          f"L_ratio={left_s2/denom2:.3f}")

    # ========================================
    # Phase3: CPU確定モード（0.035mm）で最終リファイン
    # ========================================
    print(f"\n{'='*80}")
    print(f"[Phase3] CPU確定モード(0.035mm)で最終リファインします")
    print(f"{'='*80}")
    scorer.search_mode = False  # 確定モード（閾値0.035）
    # ⚠️  biasは初期診断時に固定されているため、再測定せず全Phaseで再利用
    print(f"  🔧 GPU bias={scorer.gpu_bias:+.4f}mm（初期診断から再利用）")

    # ✅ Phase3開始前に、Phase2の最終姿勢で strict再評価（obj/score/info を統一）
    print(f"\n🔍 Phase3開始前: Phase2最終姿勢(tx={tx_best:.3f}, tz={tz_gyu:.3f})を strict で再評価...")
    obj_start, score_start, info_start = objective(
        tx_best, rx_best, ry_best, tz_gyu, scorer, 
        w_lr=1.5, w_pml=0.9, pml_margin=0.10, w_mr=0.3, 
        mode="strict"  # ★ CPU確定モードで obj を作り直す
    )
    _, comp_start = objective_from_info(
        score_start, info_start, scorer, 1.2, 0.8, 0.10, 0.4
    )
    L_ratio_start = comp_start["L_ratio"]
    pm_l_share_start = comp_start["pm_l_share"]
    
    # 🔍 デバッグ：obj計算の詳細を表示
    rs_debug = info_start["region_scores"]
    L_debug = rs_debug["M_L"] + rs_debug["PM_L"]
    R_debug = rs_debug["M_R"] + rs_debug["PM_R"]
    pen_lr_debug = abs(L_ratio_start - scorer.target_L_ratio)
    excess_debug = max(0.0, pm_l_share_start - (scorer.target_PM_L_share + 0.10))
    mr_debug = rs_debug["M_R"]
    obj_calc = score_start - 1.2 * pen_lr_debug - 0.8 * excess_debug + 0.4 * mr_debug
    print(f"  🐞 DEBUG: score={score_start:.3f}, pen_lr={pen_lr_debug:.3f}, excess={excess_debug:.3f}, mr={mr_debug:.3f}")
    print(f"  🐞 obj計算: {score_start:.3f} - 1.2*{pen_lr_debug:.3f} - 0.8*{excess_debug:.3f} + 0.4*{mr_debug:.3f} = {obj_calc:.3f}")
    
    print(f"  [STRICT phase3_start] obj={obj_start:.3f} score={score_start:.3f} "
          f"area={info_start['total_area']:.4f} contacts={info_start['num_contacts']} "
          f"L_ratio={L_ratio_start:.3f} PM_L_share={pm_l_share_start:.3f}")

    tx3, rx3, ry3, tz3, score3, info3 = hill_climb_4d(
        scorer,
        tx_init=tx_best, 
        rx_init=rx_best, 
        ry_init=ry_best, 
        tz_init=tz_gyu,
        tx_step=0.02,      # ★小さめに詰める
        deg_step=0.25,     # ★小さめに詰める
        tz_step=0.01,      # ★小さめに詰める
        max_iter=15,
        tx_min=-0.8,
        tx_max=0.8,
        max_rot_deg=5.0,
        tz_min=-2.0,
        tz_max=2.0,
        force_cpu_eval=True,  # ★これが重要：CPU確定評価
    )

    print("\nPhase3 CPU確定結果")
    print("-" * 80)
    print(f"  tx = {tx3:6.3f} mm")
    print(f"  rx = {np.rad2deg(rx3):6.3f} °")
    print(f"  ry = {np.rad2deg(ry3):6.3f} °")
    print(f"  tz = {tz3:6.3f} mm")
    print(f"  score           = {score3:.3f}")
    print(f"  total area      = {info3['total_area']:.4f} mm²")
    ra3 = info3["region_areas"]
    print(f"  M_L area        = {ra3['M_L']:.4f} mm²")
    print(f"  M_R area        = {ra3['M_R']:.4f} mm²")
    print(f"  PM_L area       = {ra3['PM_L']:.4f} mm²")
    print(f"  PM_R area       = {ra3['PM_R']:.4f} mm²")
    print(f"  ANT area        = {ra3['ANT']:.4f} mm²")
    print(f"  contacts        = {info3['num_contacts']} points")
    print(f"  spring min      = {info3['spring_min']:.4f}")
    print(f"  spring var      = {info3['spring_var']:.4f}")
    print(f"  dead springs    = {info3['spring_zero']}")
    print(f"  🔍 min_dist_raw = {info3.get('min_dist_raw', 'N/A'):.4f} mm")
    
    # ★ Phase3結果の深噛み警告
    min_dist_p3 = info3.get('min_dist_raw', 999.0)
    if min_dist_p3 < 0.005:
        print(f"  ⚠️  深噛み警告: min_dist_raw={min_dist_p3:.4f}mm < 0.005mm（めり込みリスク）")
    elif min_dist_p3 < 0.010:
        print(f"  ⚠️  注意: min_dist_raw={min_dist_p3:.4f}mm < 0.010mm（やや深い噛み込み）")
    else:
        print(f"  ✓ min_dist_raw={min_dist_p3:.4f}mm（良好）")
    
    print("-" * 80)

    rs3 = info3["region_scores"]
    print("\n  [region_scores (strength)]")
    print(f"  M_L={rs3['M_L']:.6f}, M_R={rs3['M_R']:.6f}, "
          f"PM_L={rs3['PM_L']:.6f}, PM_R={rs3['PM_R']:.6f}, ANT={rs3['ANT']:.6f}")

    left_s3  = rs3["M_L"] + rs3["PM_L"]
    right_s3 = rs3["M_R"] + rs3["PM_R"]
    denom3 = left_s3 + right_s3 + 1e-9
    print(f"  L_strength={left_s3:.6f}, R_strength={right_s3:.6f}, "
          f"L_ratio={left_s3/denom3:.3f}")

    # ========================================
    # Phase3b: CPU確定(0.035mm)で tzのみギュッ詰め
    # ========================================
    print(f"\n{'='*80}")
    print("[Phase3b] CPU確定(0.035mm) tzスキャン（全候補ログ）")
    print(f"{'='*80}")
    scorer.search_mode = False  # contact_threshold=0.035

    # Phase3の最終姿勢を起点にする
    tx0, rx0, ry0, tz0 = tx3, rx3, ry3, tz3

    # objective関数で使う重み（Phase3と同じ、M_R優勢を抑制する設定）
    w_lr = 1.5   # 1.2→1.5（左右バランス強化）
    w_pml = 0.9  # 0.8→0.9（左小臼歯バランス）
    pml_margin = 0.10
    w_mr = 0.3   # 0.4→0.3（右大臼歯報酬を減らす）

    # ベース評価（0.035mm で CPU確定）
    base_s, base_info = scorer.evaluate(tx0, rx0, ry0, tz0, force_cpu=True)
    base_obj, base_comp = objective_from_info(base_s, base_info, scorer, w_lr, w_pml, pml_margin, w_mr)
    print(f"  base tz={tz0:.3f} obj={base_obj:.3f} score={base_s:.3f} area={base_info['total_area']:.4f} "
          f"contacts={base_info['num_contacts']} dead={base_comp['dead']} spring_min={base_info['spring_min']:.4f}")

    # objective最良とscore最良を別々に追跡
    best_tz = tz0
    best_obj = base_obj
    best_s  = base_s
    best_info = base_info
    best_comp = base_comp  # ★ 追加：初期化
    
    best_score_tz = tz0
    best_score = base_s
    best_score_info = base_info
    best_score_obj = base_obj
    best_score_comp = base_comp

    # ⚡ 高速化：±0.03mm を 0.01mm刻みでGPU評価 → 上位のみCPU確定
    print("\n  🔍 Phase3b: GPU評価で候補絞り込み → 上位CPU確定（高速化）")
    print("  範囲: ±0.03mm（0.01mm刻み）、GPU評価 → 上位5個をCPU確定")
    
    # 1) GPU評価で全候補をスクリーニング（高速）
    gpu_candidates = []
    for i in range(-3, 4):  # -0.03 ... +0.03（7候補）
        dtz = i * 0.01
        cand_tz = tz0 + dtz
        s_gpu, info_gpu = scorer.evaluate(tx0, rx0, ry0, cand_tz, force_cpu=False)
        obj_gpu, comp_gpu = objective_from_info(s_gpu, info_gpu, scorer, w_lr, w_pml, pml_margin, w_mr)
        gpu_candidates.append((obj_gpu, s_gpu, cand_tz, dtz))
    
    # 2) GPU評価でobjective上位5個を選定
    gpu_candidates.sort(reverse=True, key=lambda x: x[0])
    TOP_K = 5
    print(f"  GPU評価: {len(gpu_candidates)}候補 → objective上位{TOP_K}個をCPU確定中...")
    
    # 3) 上位のみCPU確定評価
    print("\n  CPU確定結果:")
    print("  🔍 各行の意味: pen_lr=左右バランス罰, pen_pml_s=PM_L不足罰, pen_ant=前歯過多罰, pen_deep=深噛み罰, mr=右大臼歯報酬")
    print("                L_ratio=左側割合, ANT_share=前歯割合, PM_L_a=PM_L面積, min_dist=最小距離(めり込み検知)")
    
    for obj_gpu, s_gpu, cand_tz, dtz in gpu_candidates[:TOP_K]:
        s, info = scorer.evaluate(tx0, rx0, ry0, cand_tz, force_cpu=True)
        obj, comp = objective_from_info(s, info, scorer, w_lr, w_pml, pml_margin, w_mr)

        marker_obj = "★" if obj > best_obj else " "
        marker_score = "◆" if s > best_score else " "
        marker = marker_obj + marker_score
        
        print(f"  {marker} tz={cand_tz:.3f} (dtz={dtz:+.3f}) obj={obj:.3f} score={s:.3f} area={info['total_area']:.4f} "
              f"contacts={info['num_contacts']:2d} dead={comp['dead']} | "
              f"pen_lr={comp['pen_lr_effective']:.4f} pen_pml_s={comp['pen_pml_shortage']:.4f} pen_ant={comp['pen_ant']:.4f} pen_deep={comp['pen_deep']:.4f} mr={comp['mr']:.4f} | "
              f"L_ratio={comp['L_ratio']:.3f} ANT_share={comp['ANT_share']:.3f} PM_L_a={comp['PM_L_area']:.4f} min_dist={comp['min_dist_raw']:.4f}")

        if obj > best_obj:
            best_obj, best_s, best_info, best_tz = obj, s, info, cand_tz
            best_comp = comp
        
        if s > best_score:
            best_score, best_score_tz, best_score_info = s, cand_tz, info
            best_score_obj, best_score_comp = obj, comp

    print(f"\n[Phase3b] objective最良: tz={best_tz:.3f} obj={best_obj:.3f} score={best_s:.3f} area={best_info['total_area']:.4f} "
          f"contacts={best_info['num_contacts']} dead={best_info['spring_zero']} spring_min={best_info['spring_min']:.4f}")
    
    # ★ 深噛み・ANT過多・PM_L不足の警告表示
    is_watertight = getattr(scorer, 'mesh_is_watertight', True)
    critical_th = 0.005 if is_watertight else 0.010
    warning_th = 0.010 if is_watertight else 0.015
    
    if best_comp.get("deep_bite_warning", False):
        print(f"  ⚠️  深噛み警告: min_dist_raw={best_comp['min_dist_raw']:.4f}mm < {critical_th:.3f}mm（めり込みリスク）")
        print(f"      → STL水密化＋score最良（tz={best_score_tz:.3f}）の採用を検討してください")
    elif best_comp["min_dist_raw"] < warning_th:
        print(f"  ⚠️  注意: min_dist_raw={best_comp['min_dist_raw']:.4f}mm < {warning_th:.3f}mm（やや深い噛み込み）")
    else:
        print(f"  ✓ min_dist_raw={best_comp['min_dist_raw']:.4f}mm（良好）")
    
    if best_comp.get("ANT_critical", False):
        print(f"  ⚠️  前歯過多警告: ANT_share={best_comp['ANT_share']:.1%} > 40%（臼歯支持不足）")
    elif best_comp["ANT_share"] > 0.30:
        print(f"  ⚠️  注意: ANT_share={best_comp['ANT_share']:.1%} > 30%（やや前歯優位）")
    else:
        print(f"  ✓ ANT_share={best_comp['ANT_share']:.1%}（良好）")
    
    if best_comp["PM_L_area"] < 0.01:
        print(f"  ⚠️  PM_L不足警告: 面積={best_comp['PM_L_area']:.4f}mm² < 0.01mm²（左小臼歯支持不足、点数={best_comp['PM_L_count']}点）")
    else:
        print(f"  ✓ PM_L面積={best_comp['PM_L_area']:.4f}mm²、点数={best_comp['PM_L_count']}点（良好）")
    
    if abs(best_score_tz - best_tz) > 0.001:
        print(f"\n[Phase3b] score最良:     tz={best_score_tz:.3f} obj={best_score_obj:.3f} score={best_score:.3f} area={best_score_info['total_area']:.4f} "
              f"contacts={best_score_info['num_contacts']} dead={best_score_info['spring_zero']} (★比較用に別保存)")
        
        # ★ score最良が危険域の場合は却下警告
        score_is_dangerous = best_score_comp["min_dist_raw"] < 0.001  # 1µm未満は非水密の影響で不信頼
        
        if score_is_dangerous:
            print(f"  🚫 危険域判定: min_dist_raw={best_score_comp['min_dist_raw']:.4f}mm < 0.001mm（非水密STLの影響で不信頼）")
            print(f"      → score最良は保存しますが、objective最良（tz={best_tz:.3f}）の採用を強く推奨します")
        elif best_score_comp.get("deep_bite_warning", False):
            print(f"  ⚠️  深噛み警告: min_dist_raw={best_score_comp['min_dist_raw']:.4f}mm < {critical_th:.3f}mm")
        elif best_score_comp["min_dist_raw"] < warning_th:
            print(f"  ⚠️  注意: min_dist_raw={best_score_comp['min_dist_raw']:.4f}mm < {warning_th:.3f}mm")
        else:
            print(f"  ✓ min_dist_raw={best_score_comp['min_dist_raw']:.4f}mm（良好）")
        
        if best_score_comp.get("ANT_critical", False):
            print(f"  ⚠️  前歯過多警告: ANT_share={best_score_comp['ANT_share']:.1%} > 40%")
        if best_score_comp["PM_L_area"] < 0.01:
            print(f"  ⚠️  PM_L不足警告: 面積={best_score_comp['PM_L_area']:.4f}mm² < 0.01mm²")
            print(f"  ⚠️  PM_L不足警告: 接触点数={best_score_comp['PM_L_count']}点 < 2点")
    else:
        print(f"  → objective最良とscore最良が一致しています")
    
    # 詳細表示
    raF = best_info["region_areas"]
    rsF = best_info["region_scores"]
    print(f"  面積: M_L={raF['M_L']:.4f}, M_R={raF['M_R']:.4f}, PM_L={raF['PM_L']:.4f}, PM_R={raF['PM_R']:.4f}, ANT={raF['ANT']:.4f}")
    print(f"  強度: M_L={rsF['M_L']:.4f}, M_R={rsF['M_R']:.4f}, PM_L={rsF['PM_L']:.4f}, PM_R={rsF['PM_R']:.4f}, ANT={rsF['ANT']:.4f}")

    # ★ STL に反映するのは Phase3b 後の姿勢（objective最良）
    final_tx = tx0
    final_rx = rx0
    final_ry = ry0
    final_tz = best_tz
    final_ty = 0.0  # 4DOFなのでty=0
    
    # 回転中心：下顎メッシュの重心（常に下顎が動く側）
    pivot_lower = lower.vertices.mean(axis=0)
    
    # 変換行列 A を構築（下顎に適用する変換）
    A = build_transform_matrix(
        tx=final_tx,
        ty=final_ty,
        rx_rad=final_rx,
        ry_rad=final_ry,
        tz=final_tz,
        pivot=pivot_lower
    )
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 出力形式に応じてSTLを生成
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if output_mode == "lower":
        # 【下顎出力モード】下顎に A を適用
        lower_transformed = apply_transform_to_points(lower.vertices, A)
        
        output_mesh = lower.copy()
        output_mesh.vertices = lower_transformed
        
        out_dir = os.path.dirname(lower_path)
        jaw_name = os.path.splitext(os.path.basename(lower_path))[0]
        out_path = os.path.join(out_dir, f"{jaw_name}_spring5_balanced_gyu_v4.stl")
        
        output_mesh.export(out_path)
        
        print(f"\n✓ 【下顎出力】変換後STL: {out_path}")
        
    else:
        # 【上顎出力モード】上顎に A⁻¹ を適用（相対咬合は完全一致）
        A_inv = np.linalg.inv(A)
        upper_transformed = apply_transform_to_points(upper.vertices, A_inv)
        
        output_mesh = upper.copy()
        output_mesh.vertices = upper_transformed
        
        out_dir = os.path.dirname(upper_path)
        jaw_name = os.path.splitext(os.path.basename(upper_path))[0]
        out_path = os.path.join(out_dir, f"{jaw_name}_spring5_balanced_gyu_v4.stl")
        
        output_mesh.export(out_path)
        
        print(f"\n✓ 【上顎出力】変換後STL: {out_path}")
        print(f"  （下顎は元のまま、上顎を A⁻¹ で移動 → 下顎出力と相対咬合は完全一致）")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 検証：変換行列の再現性チェック
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if output_mode == "lower":
        reconstructed = apply_transform_to_points(lower.vertices, A)
        rms_error = np.sqrt(np.mean((lower_transformed - reconstructed)**2))
    else:
        reconstructed = apply_transform_to_points(upper.vertices, A_inv)
        rms_error = np.sqrt(np.mean((upper_transformed - reconstructed)**2))
    print(f"  [検証] 変換の内部一貫性: RMS誤差 = {rms_error:.6e} mm")
    if rms_error > 1e-6:
        print(f"  ⚠️  警告: RMS誤差が大きい（{rms_error:.3e} mm）")
    else:
        print(f"  ✓ 変換は正しく適用されています")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 処理時間の表示
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60
    
    print("=" * 80)
    if minutes > 0:
        print(f"⏱️  処理時間: {minutes}分 {seconds:.2f}秒")
    else:
        print(f"⏱️  処理時間: {seconds:.2f}秒")
    print("=" * 80)

def objective(tx, rx, ry, tz, scorer,
              w_lr=1.5, w_pml=0.9, pml_margin=0.10,
              w_mr=0.3, mode="search"):
    """
    目的関数（obj）を計算する
    
    mode:
      - "search": GPU候補生成/探索用（閾値0.050など、scorer.search_modeに従う）
      - "strict": CPU確定用（閾値0.035、三角形面最近接、force_cpu=True固定）
    """
    if mode == "strict":
        # ✅ CPU厳密評価を強制（Phase3/Phase3b用）
        score, info = scorer.evaluate(tx, rx, ry, tz, force_cpu=True)
    else:
        # 探索モード（scorer.search_modeに従う）
        score, info = scorer.evaluate(tx, rx, ry, tz)
    
    obj, comp = objective_from_info(score, info, scorer, w_lr, w_pml, pml_margin, w_mr)
    return obj, score, info

def gyu_refine_tz(
    scorer,
    tx, rx, ry, tz_start,
    extra_depth=0.10,
    step=0.01,                # ← 符号は後で自動決定するので正の値でOK
    closing_sign=-1,          # ← 閉口方向（診断から自動判定）
    max_score_drop=0.11,
    # ★ Phase3と同じobjective関数の重み（M_R優勢抑制）
    w_lr=1.5,          # 1.2→1.5（左右バランス強化）
    w_pml=0.9,         # 0.8→0.9
    pml_margin=0.10,
    w_mr=0.3,          # 0.4→0.3（右大臼歯報酬減）
):
    print("\n[Phase2: gyu_refine_tz] 2段階評価（軽い絞込→厳密決定）")
    
    # 🔍 検査ログ: Phase2開始時の姿勢を記録
    print(f"\n[POSE phase2_base] tx={tx:.3f} rx={np.rad2deg(rx):.3f}° ry={np.rad2deg(ry):.3f}° tz={tz_start:.3f}")
    
    # ✅ base を厳密評価（search_mode=False, 0.035mm）
    base_score, base_info = scorer.evaluate(tx, rx, ry, tz_start, force_cpu=True)
    
    # 🔍 検査ログ: Phase2のbase評価結果を記録
    print(f"[STRICT phase2_base] score={base_score:.3f} area={base_info['total_area']:.4f} contacts={base_info['num_contacts']} dead={base_info['spring_zero']}")
    ra_base = base_info["region_areas"]
    print(f"  area_by_region: M_L={ra_base['M_L']:.4f} M_R={ra_base['M_R']:.4f} PM_L={ra_base['PM_L']:.4f} PM_R={ra_base['PM_R']:.4f} ANT={ra_base['ANT']:.4f}")
    
    print(f"\n  base tz={tz_start:.3f} score={base_score:.3f} area={base_info['total_area']:.4f} "
          f"contacts={base_info['num_contacts']} dead={base_info['spring_zero']}")

    # ★閉口方向は診断から自動判定された closing_sign を使用
    step = closing_sign * abs(step)
    tz_limit = tz_start + closing_sign * extra_depth
    direction_str = "tz+" if closing_sign > 0 else "tz-"
    print(f"  → 閉口方向: {direction_str} (limit={tz_limit:.3f}) [診断から自動判定]")

    # ★Phase3と同じobjectiveで評価
    base_obj, base_comp = objective_from_info(base_score, base_info, scorer, w_lr, w_pml, pml_margin, w_mr)
    print(f"  base objective={base_obj:.3f} (pen_lr={base_comp['pen_lr']:.4f}, excess={base_comp['excess']:.4f}, mr={base_comp['mr']:.4f})")

    best_tz, best_score, best_info, best_obj = tz_start, base_score, base_info, base_obj

    tz = tz_start + step
    print(f"\n  候補スキャン（2段階評価）:")
    candidates = []  # 厳密評価待ちリスト
    
    while (tz <= tz_limit + 1e-9) if step > 0 else (tz >= tz_limit - 1e-9):
        # ✅ Step1: 軽い評価で絞り込み（0.040mm探索閾値、速い）
        scorer.search_mode = True
        score_loose, info_loose = scorer.evaluate(tx, rx, ry, tz)
        scorer.search_mode = False
        
        # 明らかにダメな候補は除外（緩い基準：5点以上で候補に）
        if info_loose["num_contacts"] < 5:  # 8→5に緩和
            print(f"    tz={tz:.3f} SKIP (loose contacts={info_loose['num_contacts']} < 5)")
            tz += step
            continue
        
        # ✅ Step2: 厳密評価（0.035mm CPU, Phase3と完全同一）
        score, info = scorer.evaluate(tx, rx, ry, tz, force_cpu=True)
        
        # 厳密基準で除外
        if info["spring_zero"] > 0:
            print(f"    tz={tz:.3f} SKIP (dead_springs={info['spring_zero']})")
            tz += step
            continue

        if info["num_contacts"] < 10:
            print(f"    tz={tz:.3f} SKIP (strict contacts={info['num_contacts']} < 10)")
            tz += step
            continue

        # objective 計算（★Phase3と完全同一の評価関数）
        if score >= base_score - max_score_drop:
            obj, comp = objective_from_info(score, info, scorer, w_lr, w_pml, pml_margin, w_mr)
            
            if obj > best_obj:
                print(f"  ★ tz={tz:.3f} obj={obj:.3f} score={score:.3f} area={info['total_area']:.4f} "
                      f"contacts={info['num_contacts']} dead={info['spring_zero']} | "
                      f"pen_lr={comp['pen_lr']:.4f} excess={comp['excess']:.4f} mr={comp['mr']:.4f} (IMPROVED)")
                best_obj, best_tz, best_score, best_info = obj, tz, score, info
            else:
                print(f"    tz={tz:.3f} obj={obj:.3f} score={score:.3f} area={info['total_area']:.4f} "
                      f"contacts={info['num_contacts']} | pen_lr={comp['pen_lr']:.4f}")

        tz += step

    print(f"\n[Phase2] 最終結果（CPU厳密 0.035mm）: tz={best_tz:.3f} obj={best_obj:.3f} score={best_score:.3f} "
          f"area={best_info['total_area']:.4f} contacts={best_info['num_contacts']} dead={best_info['spring_zero']}")
    return best_tz, best_score, best_info


if __name__ == "__main__":
    main()


import os
import sys
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
# ユーティリティ
# =============================

def select_two_stl_files():
    """
    ファイルダイアログから STL ファイルを1顎ずつ選択
    1回目: 上顎, 2回目: 下顎
    """
    root = Tk()
    root.withdraw()
    
    print("\n=== STLファイル選択 ===")
    print("まず上顎のSTLファイルを選択してください...")
    input("Enterキーを押してファイル選択を開始...")

    upper_path = filedialog.askopenfilename(
        title="🦷 上顎（Upper Jaw）のSTLファイルを選択 - STEP 1/2",
        filetypes=[("STL files", "*.stl"), ("All files", "*.*")]
    )
    if not upper_path:
        print("❌ エラー: 上顎 STL が選択されませんでした。")
        root.destroy()
        sys.exit(1)
    
    print(f"✓ 上顎選択完了: {os.path.basename(upper_path)}")
    print("\n次に下顎のSTLファイルを選択してください...")
    input("Enterキーを押して下顎選択を開始...")

    lower_path = filedialog.askopenfilename(
        title="🦷 下顎（Lower Jaw）のSTLファイルを選択 - STEP 2/2",
        filetypes=[("STL files", "*.stl"), ("All files", "*.*")]
    )
    if not lower_path:
        print("❌ エラー: 下顎 STL が選択されませんでした。")
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
        if not mesh.is_watertight:
            print(f"警告: {os.path.basename(filepath)} は水密ではありません")
        if len(mesh.vertices) < 100:
            raise ValueError(f"頂点数が少なすぎます: {len(mesh.vertices)}")
        print(f"✓ {os.path.basename(filepath)} 読み込み成功 ({len(mesh.vertices)} 頂点)")
        return mesh
    except Exception as e:
        print(f"エラー: {filepath} の読み込みに失敗しました")
        print("詳細:", e)
        sys.exit(1)


def per_vertex_area(mesh: trimesh.Trimesh):
    """
    各三角形の面積を3頂点に等分配して頂点面積とする
    """
    areas = np.zeros(len(mesh.vertices))
    for face, area in zip(mesh.faces, mesh.area_faces):
        for vid in face:
            areas[vid] += area / 3.0
    return areas


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
    ):
        self.upper = upper_mesh
        self.v0 = lower_sample_vertices  # 下顎サンプル頂点（基準座標）
        self.areas = lower_sample_areas
        self.contact_threshold = contact_threshold
        self.rot_penalty = rot_penalty
        self.trans_penalty = trans_penalty
        
        # GPU加速用データの準備
        if GPU_AVAILABLE:
            # メモリ効率を考慮してfloat32使用
            self.v0_gpu = array_to_gpu(self.v0.astype(np.float32))
            self.areas_gpu = array_to_gpu(self.areas.astype(np.float32))
            self.upper_vertices_gpu = array_to_gpu(upper_mesh.vertices.astype(np.float32))
            
            # GPUメモリ使用量を表示
            gpu_memory_mb = (
                self.v0_gpu.nbytes + self.areas_gpu.nbytes + 
                self.upper_vertices_gpu.nbytes
            ) / (1024 * 1024)
            
            print(f"✓ GPU メモリに転送完了: {len(self.v0)} 下顎頂点, {len(upper_mesh.vertices)} 上顎頂点")
            print(f"✓ GPU メモリ使用量: {gpu_memory_mb:.1f} MB")
            
            # メモリ使用量チェック
            if hasattr(cp, 'get_default_memory_pool'):
                mempool = cp.get_default_memory_pool()
                print(f"✓ GPU メモリプール: {mempool.used_bytes()/(1024*1024):.1f} MB 使用中")
        else:
            self.v0_gpu = self.v0
            self.areas_gpu = self.areas

        
        # ----------------------------
        # 5ブロックへの自動分割
        # ----------------------------
        x = self.v0[:, 0]
        y = self.v0[:, 1]

        self.x_mid = float(np.median(x))
        y_min, y_max = float(y.min()), float(y.max())
        if y_max == y_min:
            # 万一全て同じ値なら、全部「臼歯」として扱う
            y_cut1 = y_min - 0.1
            y_cut2 = y_min + 0.1
        else:
            dy = y_max - y_min
            y_cut1 = y_min + dy / 3.0        # 大臼歯 / 小臼歯の境
            y_cut2 = y_min + dy * 2.0 / 3.0  # 小臼歯 / 前歯の境

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

        # 実際に頂点が存在するブロックだけを「有効バネ」とみなす
        self.valid_regions = [
            name for name, m in self.region_masks.items() if np.any(m)
        ]

        print("\n[ブロック分割（輪ゴム5本）]")
        for name in ["M_L", "M_R", "PM_L", "PM_R", "ANT"]:
            cnt = int(self.region_masks[name].sum())
            flag = "✓" if name in self.valid_regions else "（頂点なし）"
            print(f"  {name:5s}: {cnt:4d} 点 {flag}")
        print(f"  有効バネ本数: {len(self.valid_regions)}")

        eps = 1e-12
        self.region_cap = {}
        for name, mask in self.region_masks.items():
            cap = float(self.areas[mask].sum()) if np.any(mask) else 0.0
            self.region_cap[name] = cap

        capL = self.region_cap["M_L"] + self.region_cap["PM_L"]
        capR = self.region_cap["M_R"] + self.region_cap["PM_R"]
        self.target_L_ratio = capL / (capL + capR + eps)

        # 左側の中で PM_L が占める“自然な比率”（欠損でM_Lが少ないとここが上がる）
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
        """GPU加速版最近接距離計算"""
        if not GPU_AVAILABLE:
            # CPU fallback
            transformed_cpu = array_to_cpu(transformed_vertices_gpu)
            _, distances, _ = self.upper.nearest.on_surface(transformed_cpu)
            return cp.asarray(distances) if GPU_AVAILABLE else np.asarray(distances)
        
        # GPU版：最適化された最短距離計算
        n_lower = transformed_vertices_gpu.shape[0]
        n_upper = self.upper_vertices_gpu.shape[0]
        
        # メモリ効率を考慮した適応的バッチサイズ
        available_memory = 1024 * 1024 * 1024  # 1GB と仮定
        element_size = 4  # float32
        matrix_elements = n_lower * n_upper * 3
        required_memory = matrix_elements * element_size
        
        if required_memory > available_memory:
            # 大きなメッシュ：高速並列処理
            batch_size = min(400, n_lower)  # より大きなバッチで高速化
            if not hasattr(self, '_gpu_mode_notified'):
                print(f"🚀 GPU高速並列計算: バッチサイズ {batch_size}")
                self._gpu_mode_notified = True
        else:
            # 小さなメッシュ：高速処理
            batch_size = min(1000, n_lower)
        
        distances = cp.zeros(n_lower, dtype=cp.float32)
        
        for i in range(0, n_lower, batch_size):
            end_idx = min(i + batch_size, n_lower)
            batch_lower = transformed_vertices_gpu[i:end_idx]  # (batch_size, 3)
            
            # より効率的な距離計算：Broadcasting を活用
            # batch_lower: (batch_size, 3)
            # upper_vertices_gpu: (n_upper, 3)
            
            # cdist的な計算をGPUで効率的に実行
            batch_lower_expanded = batch_lower[:, cp.newaxis, :]  # (batch_size, 1, 3)
            upper_expanded = self.upper_vertices_gpu[cp.newaxis, :, :]  # (1, n_upper, 3)
            
            # 効率的なユークリッド距離計算
            diff = batch_lower_expanded - upper_expanded  # Broadcasting
            distances_sq = cp.sum(diff * diff, axis=2)  # 平方根計算を遅延
            batch_distances = cp.sqrt(distances_sq)
            
            # 各下顎頂点の最短距離
            distances[i:end_idx] = cp.min(batch_distances, axis=1)
        
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
        ※ evaluate() のように距離クリップしない（絶対当たらない判定に重要）
        """
        ty = 0.0
        rot = R.from_euler("xyz", [rx_rad, ry_rad, 0.0]).as_matrix()
        transformed = (rot @ self.v0.T).T + np.array([tx, ty, tz])

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

    # ----------------------------
    # 姿勢評価
    # ----------------------------

    def evaluate(self, tx, rx_rad, ry_rad, tz, max_dist_clip=0.05):
        """
        姿勢 (tx, rx, ry, tz) に対するスコアを返す（GPU加速版）
        - tx: 左右方向スライド（mm）
        - rx, ry: ラジアン（X, Y軸まわりの回転）
        - tz: 垂直方向（mm）
        ty は 0 固定（前後スライドはここでは見ない）

        戻り値:
          score, info_dict
        """
        ty = 0.0

        if GPU_AVAILABLE:
            # 完全GPU加速版：変換と距離計算を全てGPUで実行
            rot = R.from_euler("xyz", [rx_rad, ry_rad, 0.0]).as_matrix()
            rot_gpu = array_to_gpu(rot.astype(np.float32))
            
            # 下顎頂点をGPUで変換: v' = R @ v + t
            transformed_gpu = cp.dot(self.v0_gpu, rot_gpu.T) + cp.array([tx, ty, tz], dtype=cp.float32)
            
            # GPU完全距離計算
            distances_gpu = self._gpu_nearest_distances(transformed_gpu)
            d_gpu = cp.clip(distances_gpu, 0.0, max_dist_clip)
            contact_mask_gpu = d_gpu <= self.contact_threshold
            contact_count = int(array_to_cpu(cp.sum(contact_mask_gpu)))
            
            # *** 🔍 CRITICAL DEBUG: 生の最短距離を記録 ***
            min_dist_raw = float(array_to_cpu(cp.min(distances_gpu)))
            
            if not hasattr(self, '_gpu_calc_notified'):
                if hasattr(cp, 'get_default_memory_pool'):
                    mempool = cp.get_default_memory_pool()
                    print(f"   GPU使用中: {mempool.used_bytes()/(1024*1024):.1f} MB")
                self._gpu_calc_notified = True
        else:
            # CPU fallback
            rot = R.from_euler("xyz", [rx_rad, ry_rad, 0.0]).as_matrix()
            transformed = (rot @ self.v0.T).T + np.array([tx, ty, tz])
            _, distances, _ = self.upper.nearest.on_surface(transformed)
            d_gpu = np.clip(distances, 0.0, max_dist_clip)
            contact_mask_gpu = d_gpu <= self.contact_threshold
            contact_count = int(np.sum(contact_mask_gpu))
            
            # *** 🔍 CRITICAL DEBUG: 生の最短距離を記録 ***
            min_dist_raw = float(np.min(distances))

        # --------------------------------------------------
        # 1) まったく噛んでいない場合
        #    → 回転・移動ペナルティ + 大きなマイナス定数
        #       （どんな「噛んでいる姿勢」より必ず不利にする）
        # --------------------------------------------------
        if contact_count == 0:
            rot_pen = self.rot_penalty * (abs(rx_rad) + abs(ry_rad))
            trans_pen = self.trans_penalty * np.sqrt(tx * tx + tz * tz)

            # 「接触ゼロは最低でも -10 点」くらいにしておく
            score = -(rot_pen + trans_pen) - 10.0

            zero_dict = {name: 0.0 for name in self.region_masks.keys()}
            info = {
                "total_area": 0.0,
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
        # 2) ここから下は「接触あり」のケース（GPU加速版）
        # --------------------------------------------------

        # contact_mask 部だけの距離・面積
        th = self.contact_threshold
        d_c_gpu = d_gpu[contact_mask_gpu]
        w_gpu = 1.0 - (d_c_gpu / th) ** 2               # d=0 で1, d=th で0
        w_gpu = cp.clip(w_gpu, 0.0, 1.0)

        # 「バネの縮み量 × 断面積」のようなイメージ
        local_strength_c_gpu = self.areas_gpu[contact_mask_gpu] * w_gpu

        # 全頂点長の配列に戻す（コンタクト頂点以外は0）
        strength_full_gpu = cp.zeros_like(self.areas_gpu)
        area_full_gpu = cp.zeros_like(self.areas_gpu)
        strength_full_gpu[contact_mask_gpu] = local_strength_c_gpu
        area_full_gpu[contact_mask_gpu] = self.areas_gpu[contact_mask_gpu]

        # ----- バネごとのスコア・面積（GPU版） -----
        region_scores = {}
        region_areas = {}
        scores_list = []

        for name in self.valid_regions:
            mask = self.region_masks[name]
            # GPU配列からCPUマスクを適用
            if GPU_AVAILABLE:
                mask_gpu = array_to_gpu(mask)
                s = float(array_to_cpu(cp.sum(strength_full_gpu[mask_gpu])))
                a = float(array_to_cpu(cp.sum(area_full_gpu[mask_gpu])))
            else:
                s = float(strength_full_gpu[mask].sum())
                a = float(area_full_gpu[mask].sum())
            
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
        if GPU_AVAILABLE:
            total_area = float(array_to_cpu(cp.sum(area_full_gpu)))
        else:
            total_area = float(area_full_gpu.sum())

        # 5本の輪ゴムの状態
        if len(scores_arr) > 0:
            min_region = float(scores_arr.min())
            var_region = float(scores_arr.var())
            mean_region = float(scores_arr.mean())
            zero_regions = int(np.sum(scores_arr < 1e-6))
        else:
            min_region = 0.0
            var_region = 0.0
            mean_region = 0.0
            zero_regions = 0

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

        info = {
            "total_area": total_area,
            "num_contacts": int(array_to_cpu(contact_count)),
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


# =============================
# 探索アルゴリズム
# =============================

def line_search_tz(scorer: SpringOcclusionScorer,
                   tx0=0.0, rx0=0.0, ry0=0.0,
                   tz_start=0.5, tz_end=-1.5, step=-0.05,
                   # ★バランス補正の重み（まずはこのくらいから）
                   w_lr=1.2,          # 左右バランス（L_ratio vs target_L_ratio）
                   w_pml=0.8,         # 左小臼歯（PM_L）の偏り抑制
                   pml_margin=0.10,   # “許容する”PM_L share の余裕
                   w_mr=0.4           # 右大臼歯（M_R）を少し押す
                   ):
    """
    tz 方向にまっすぐ閉口しながら、
    score最大ではなく「score + バランス補正」を最大化する tz を探す
    → これをヒルクライムの初期値にする
    """

    def objective(tx, rx, ry, tz):
        score, info = scorer.evaluate(tx, rx, ry, tz)
        rs = info["region_scores"]

        L = rs["M_L"] + rs["PM_L"]
        R = rs["M_R"] + rs["PM_R"]
        denom = L + R + 1e-12
        L_ratio = L / denom

        # 左側の中でPM_Lが占める比率（左大臼歯欠損などで上がりやすい）
        pm_l_share = rs["PM_L"] / (L + 1e-12)

        # 目標からのズレ（欠損を cap で見て target_L_ratio が決まる想定）
        pen_lr = abs(L_ratio - scorer.target_L_ratio)

        # PM_L偏りが「自然比 + margin」を超えた分だけ抑える
        excess = max(0.0, pm_l_share - (scorer.target_PM_L_share + pml_margin))
        pen_pml = excess

        # 右大臼歯を少し押す（“見た目で右が弱い”対策）
        mr = rs["M_R"]

        obj = score - w_lr * pen_lr - w_pml * pen_pml + w_mr * mr
        return obj, score, info, L_ratio, pm_l_share

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
            print(
                f"  tz={tz:6.3f} mm -> obj={obj:7.3f}, score={score:7.3f}, "
                f"area={info['total_area']:.4f}, min_dist={min_raw:.4f}mm | "
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
    return best_tz, best_score, best_info


def hill_climb_4d(scorer: SpringOcclusionScorer,
                  tx_init, rx_init, ry_init, tz_init,
                  tx_step=0.05, deg_step=0.5, tz_step=0.05,
                  max_iter=20,
                  tx_min=-0.8, tx_max=0.8,
                  max_rot_deg=5.0,
                  tz_min=-2.0, tz_max=1.0,
                  # ★バランス補正の重み（まずはこのくらいから）
                  w_lr=1.2,          # 左右バランス（L_ratio vs target_L_ratio）
                  w_pml=0.8,         # 左小臼歯（PM_L）偏り抑制
                  pml_margin=0.10,   # PM_L share “許容マージン”
                  w_mr=0.4           # 右大臼歯（M_R）を少し押す
                  ):
    """
    (tx, rx, ry, tz) の4自由度ヒルクライム
    ただし比較は score ではなく objective（score + バランス補正）で行う
    """

    def objective(tx, rx, ry, tz):
        score, info = scorer.evaluate(tx, rx, ry, tz)
        rs = info["region_scores"]

        L = rs["M_L"] + rs["PM_L"]
        R = rs["M_R"] + rs["PM_R"]
        denom = L + R + 1e-12
        L_ratio = L / denom

        pm_l_share = rs["PM_L"] / (L + 1e-12)

        # 目標からのズレ
        pen_lr = abs(L_ratio - scorer.target_L_ratio)

        # PM_L偏り：自然比 + margin を超えた分だけ抑える
        excess = max(0.0, pm_l_share - (scorer.target_PM_L_share + pml_margin))
        pen_pml = excess

        # 右大臼歯（見た目で右が弱い対策）
        mr = rs["M_R"]

        obj = score - w_lr * pen_lr - w_pml * pen_pml + w_mr * mr
        return obj, score, info, L_ratio, pm_l_share

    tx = tx_init
    rx = rx_init
    ry = ry_init
    tz = tz_init

    obj, score, info, L_ratio, pm_l_share = objective(tx, rx, ry, tz)
    print("\n[Step2] 近傍ヒルクライム開始（objective で最適化）")
    print(
        f"  start: tx={tx:.3f}mm, rx={np.rad2deg(rx):.3f}°, "
        f"ry={np.rad2deg(ry):.3f}°, tz={tz:.3f} mm, "
        f"obj={obj:.3f}, score={score:.3f}, area={info['total_area']:.4f}, "
        f"L_ratio={L_ratio:.3f}, PM_L_share={pm_l_share:.3f}"
    )

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
        
        if neighbor_poses:
            # バッチ評価でGPU加速
            neighbor_poses = np.array(neighbor_poses)
            if GPU_AVAILABLE and len(neighbor_poses) > 2:  # より積極的にGPUバッチ評価を使用
                batch_scores, batch_infos = scorer.evaluate_batch(neighbor_poses)
                
                for i, (pose, score_c, info_c) in enumerate(zip(neighbor_poses, batch_scores, batch_infos)):
                    tx_c, rx_c, ry_c, tz_c = pose
                    obj_c, _, _, lr_c, pml_c = objective(tx_c, rx_c, ry_c, tz_c)
                    
                    if obj_c > best_local_obj:
                        best_local_obj = obj_c
                        best_local = (tx_c, rx_c, ry_c, tz_c)
                        best_local_score = score_c
                        best_local_info = info_c
                        best_lr = lr_c
                        best_pml = pml_c
                        improved = True
            else:
                # CPU fallback または少数候補の場合
                for pose in neighbor_poses:
                    tx_c, rx_c, ry_c, tz_c = pose
                    obj_c, score_c, info_c, lr_c, pml_c = objective(tx_c, rx_c, ry_c, tz_c)

                    if obj_c > best_local_obj:
                        best_local_obj = obj_c
                        best_local = (tx_c, rx_c, ry_c, tz_c)
                        best_local_score = score_c
                        best_local_info = info_c
                        best_lr = lr_c
                        best_pml = pml_c
                        improved = True

        if not improved:
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

    # 返す score/info は “純 score” のもの（従来互換）
    return tx, rx, ry, tz, score, info


# =============================
# メイン
# =============================

def main():
    print("=" * 80)
    print("咬頭嵌合位自動最適化（5本の輪ゴムスプリングモデル）")
    print("=" * 80)

    upper_path, lower_path = select_two_stl_files()
    upper = load_mesh_safely(upper_path)
    lower = load_mesh_safely(lower_path)

    # 下顎頂点のエリア & サンプリング
    print("\n頂点面積を計算中...")
    lower_vertex_area_all = per_vertex_area(lower)

    all_vertices = lower.vertices
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
    sample_areas = lower_vertex_area_all[sample_idx]

    # スコアラー準備
    scorer = SpringOcclusionScorer(
        upper_mesh=upper,
        lower_sample_vertices=sample_vertices,
        lower_sample_areas=sample_areas,
        contact_threshold=0.035,  # 右側窩嵌合改善のため微増（0.035mm）
        rot_penalty=1.5,
        trans_penalty=2.0,
    )

    # まず、メッシュの大まかな位置関係を調査
    print("\n[診断] メッシュ位置関係の調査")
    upper_center = upper.vertices.mean(axis=0)
    lower_center = lower.vertices.mean(axis=0)
    print(f"上顎中心: {upper_center}")
    print(f"下顎中心: {lower_center}")
    print(f"初期ギャップ: {upper_center - lower_center}")
    
    # より広い範囲での距離調査
    test_positions = [
        (0.0, 0.0, 0.0, 2.0),   # 大きく離れた位置
        (0.0, 0.0, 0.0, 1.0),   # 中間位置  
        (0.0, 0.0, 0.0, 0.0),   # 基準位置
        (0.0, 0.0, 0.0, -1.0),  # 接近位置
        (0.0, 0.0, 0.0, -2.0),  # さらに接近
    ]
    
    print("\n[診断] 各位置での最短距離調査:")
    for tx, rx, ry, tz in test_positions:
        gap_info = scorer.region_gap_info(tx, rx, ry, tz, near_th=0.5)  # 0.5mm以内
        min_distances = [info["min"] for info in gap_info.values()]
        overall_min = min(min_distances)
        print(f"  tz={tz:5.1f}mm: 最短距離={overall_min:.3f}mm")
        if overall_min < 0.1:  # 0.1mm以内なら詳細表示
            for region, info in gap_info.items():
                if info["min"] < 0.1:
                    print(f"    {region}: min={info['min']:.3f}mm, near_count={info['near_count']}")

    # Step1: tz 方向スキャンで初期位置（診断結果：tz=1.0mm付近が最適）
    best_tz, best_score_tz, info_tz = line_search_tz(
        scorer,
        tx0=0.0,
        rx0=0.0,
        ry0=0.0,
        tz_start=1.5,    # 診断で確認された最適範囲をカバー
        tz_end=0.5,      # tz=1.0mm付近を含む範囲
        step=-0.05
    )

    # Step2 (Phase1): 近傍ヒルクライム（tx も含めて最適化）
    tx_best, rx_best, ry_best, tz_best, score_best, info_best = hill_climb_4d(
        scorer,
        tx_init=0.0,
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
        tz_max=2.0,  # 診断結果のtz=1.0mm付近をカバー
    )

    print("\nPhase1 結果（ノーマル咬合位置）")
    print("-" * 80)
    print(f"  tx = {tx_best:6.3f} mm")
    print(f"  rx = {np.rad2deg(rx_best):6.3f} °")
    print(f"  ry = {np.rad2deg(ry_best):6.3f} °")
    print(f"  tz = {tz_best:6.3f} mm")
    print(f"  score           = {score_best:.3f}")
    print(f"  total area      = {info_best['total_area']:.4f} mm²")
    ra = info_best["region_areas"]
    print(f"  M_L area        = {ra['M_L']:.4f} mm²")
    print(f"  M_R area        = {ra['M_R']:.4f} mm²")
    print(f"  PM_L area       = {ra['PM_L']:.4f} mm²")
    print(f"  PM_R area       = {ra['PM_R']:.4f} mm²")
    print(f"  ANT area        = {ra['ANT']:.4f} mm²")
    print(f"  contacts        = {info_best['num_contacts']} points")
    print(f"  spring min      = {info_best['spring_min']:.4f}")
    print(f"  spring var      = {info_best['spring_var']:.4f}")
    print(f"  dead springs    = {info_best['spring_zero']}")
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

    # ★ Phase2: tz だけを少し「ギュッ」と噛み込ませる
    tz_gyu, score_gyu, info_gyu = gyu_refine_tz(
        scorer,
        tx_best, rx_best, ry_best, tz_best,
        extra_depth=0.10,  # ← ギュッとする最大量（mm）。0.05〜0.10 あたりから調整
        step=-0.01,        # 0.01mm 刻み
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

    # ★ STL に反映するのは Phase2 後の姿勢
    final_tx = tx_best
    final_rx = rx_best
    final_ry = ry_best
    final_tz = tz_gyu

    rot_final = R.from_euler("xyz", [final_rx, final_ry, 0.0]).as_matrix()
    transformed_all = (rot_final @ lower.vertices.T).T + np.array([final_tx, 0.0, final_tz])

    lower_out = lower.copy()
    lower_out.vertices = transformed_all

    out_dir = os.path.dirname(lower_path)
    lower_name = os.path.splitext(os.path.basename(lower_path))[0]
    out_path = os.path.join(out_dir, f"{lower_name}_spring5_balanced_gyu.stl")  # ★ファイル名も分けておく
    lower_out.export(out_path)
    print(f"\n✓ 最終下顎 STL を保存しました: {out_path}")
    print("=" * 80)

def objective(tx, rx, ry, tz, scorer,
              w_lr=1.2, w_pml=0.8, pml_margin=0.10,
              w_mr=0.4):
    score, info = scorer.evaluate(tx, rx, ry, tz)
    rs = info["region_scores"]

    L = rs["M_L"] + rs["PM_L"]
    R = rs["M_R"] + rs["PM_R"]
    denom = L + R + 1e-12
    L_ratio = L / denom

    pm_l_share = rs["PM_L"] / (L + 1e-12)

    pen_lr = abs(L_ratio - scorer.target_L_ratio)

    excess = max(0.0, pm_l_share - (scorer.target_PM_L_share + pml_margin))
    pen_pml = excess

    mr = rs["M_R"]

    obj = score - w_lr * pen_lr - w_pml * pen_pml + w_mr * mr
    return obj, score, info

def gyu_refine_tz(
    scorer,
    tx, rx, ry, tz_start,
    extra_depth=0.10,
    step=0.01,                # ← 符号は後で自動決定するので正の値でOK
    max_score_drop=0.11,
    w_right_post=0.60,
    w_pml_pen=0.40,
    w_depth=0.05,
):
    base_score, base_info = scorer.evaluate(tx, rx, ry, tz_start)

    # ★閉口方向（接触が増える方向）を自動判定
    delta = abs(step)
    s_plus,  i_plus  = scorer.evaluate(tx, rx, ry, tz_start + delta)
    s_minus, i_minus = scorer.evaluate(tx, rx, ry, tz_start - delta)

    # どちらが「接触が増える」かを total_area で判定（scoreでも良い）
    if i_plus["total_area"] >= i_minus["total_area"]:
        step = +delta
        tz_limit = tz_start + extra_depth
    else:
        step = -delta
        tz_limit = tz_start - extra_depth

    def S(info, key):
        return float(info["region_scores"].get(key, 0.0))

    right_post0 = S(base_info, "M_R") + S(base_info, "PM_R")
    pml0 = S(base_info, "PM_L")
    obj0 = base_score + w_right_post * right_post0 - w_pml_pen * pml0

    best_tz, best_score, best_info, best_obj = tz_start, base_score, base_info, obj0

    tz = tz_start + step
    print("\n[Phase2: gyu_refine_tz] tz を『噛み込む方向』へ探索します")
    while (tz <= tz_limit + 1e-9) if step > 0 else (tz >= tz_limit - 1e-9):
        score, info = scorer.evaluate(tx, rx, ry, tz)

        if score >= base_score - max_score_drop:
            right_post = S(info, "M_R") + S(info, "PM_R")
            pml = S(info, "PM_L")
            depth_bonus = abs(tz - tz_start)

            obj = score + w_right_post * right_post - w_pml_pen * pml + w_depth * depth_bonus
            if obj > best_obj:
                best_obj, best_tz, best_score, best_info = obj, tz, score, info

        tz += step

    print(f"\n  → gyu 結果: tz={best_tz:.3f} mm, score={best_score:.3f}")
    return best_tz, best_score, best_info

   

if __name__ == "__main__":
    main()



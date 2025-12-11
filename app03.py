import os
import sys
import numpy as np
import trimesh
from tkinter import Tk, filedialog
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from datetime import datetime
import warnings
import logging
warnings.filterwarnings('ignore')

# GPU対応の確認
try:
    import cupy as cp
    # CuPy インストール確認
    _ = cp.__version__
    GPU_AVAILABLE = cp.cuda.is_available()
    
    if GPU_AVAILABLE:
        # cuBLAS動作テスト
        try:
            test = cp.array([1.0, 2.0, 3.0])
            _ = cp.linalg.norm(test)
            print("🚀 GPU (CUDA 13.1) が利用可能です")
            print(f"   CuPy バージョン: {cp.__version__}")
        except Exception as e:
            print(f"⚠ GPU初期化エラー、CPUモードで実行します: {str(e)[:60]}...")
            GPU_AVAILABLE = False
    else:
        print("💻 CUDA デバイスが見つかりません")
        GPU_AVAILABLE = False
        
except ImportError as e:
    GPU_AVAILABLE = False
    if "cupy" in str(e):
        print("💻 CPU モードで実行します (CuPy未検出)")
        print("   ※ 仮想環境が正しく有効化されているか確認してください")
    else:
        print(f"💻 CPU モードで実行します: {e}")

try:
    import torch
    if hasattr(torch, 'cuda') and torch.cuda.is_available():
        TORCH_GPU_AVAILABLE = True
        print(f"🔥 PyTorch GPU: {torch.cuda.get_device_name(0)}")
    else:
        TORCH_GPU_AVAILABLE = False
except (ImportError, AttributeError):
    TORCH_GPU_AVAILABLE = False

# ログ設定
def setup_logging():
    """ログ設定を初期化"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('optimization.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


# ========= ここで可視化のON/OFFを切り替え =========
ENABLE_PLOT = False   # True にすると candidates_comparison.png を生成
ENABLE_HTML = True    # False にすると HTML レポートを生成しない
# =================================================


class OptimizationConfig:
    """最適化パラメータの設定クラス"""
    
    # プリセット定義
    PRESETS = {
        "バランス型": {
            "CONTACT_THRESHOLD": 0.025,      # 接触判定を緩和（より多くの接触点を検出）
            "TIGHT_THRESHOLD": 0.008,        # 緊密接触の範囲を拡大
            "BALANCE_AP_WEIGHT": 0.6,        # 前後バランスを重視
            "BALANCE_LR_WEIGHT": 0.6,        # 左右バランスを重視
            "PENETRATION_PENALTY": 0.3,      # めり込みペナルティを軽減
            "ROTATION_PENALTY": 1.5,         # 回転制限を緩和
            "TRANSLATION_PENALTY": 2.5,      # 移動制限を緩和
        },
        "前歯部優位": {
            "CONTACT_THRESHOLD": 0.02,
            "TIGHT_THRESHOLD": 0.005,
            "BALANCE_AP_WEIGHT": 0.6,  # 前後を重視
            "BALANCE_LR_WEIGHT": 0.2,
            "PENETRATION_PENALTY": 0.5,
            "ROTATION_PENALTY": 2.0,
            "TRANSLATION_PENALTY": 3.0,
        },
        "臼歯部優位": {
            "CONTACT_THRESHOLD": 0.025,
            "TIGHT_THRESHOLD": 0.005,
            "BALANCE_AP_WEIGHT": 0.2,
            "BALANCE_LR_WEIGHT": 0.5,  # 左右を重視
            "PENETRATION_PENALTY": 0.3,
            "ROTATION_PENALTY": 1.5,
            "TRANSLATION_PENALTY": 2.0,
        },
        "緊密接触型": {
            "CONTACT_THRESHOLD": 0.015,  # より厳しい閾値
            "TIGHT_THRESHOLD": 0.003,
            "BALANCE_AP_WEIGHT": 0.3,
            "BALANCE_LR_WEIGHT": 0.3,
            "PENETRATION_PENALTY": 0.8,  # めり込みを強く抑制
            "ROTATION_PENALTY": 2.5,
            "TRANSLATION_PENALTY": 3.5,
        },
        # ★ 追加：高速テスト用（時間短縮用プリセット）
        "高速テスト用": {
            "CONTACT_THRESHOLD": 0.02,
            "TIGHT_THRESHOLD": 0.005,
            "BALANCE_AP_WEIGHT": 0.4,
            "BALANCE_LR_WEIGHT": 0.4,
            "PENETRATION_PENALTY": 0.5,
            "ROTATION_PENALTY": 2.0,
            "TRANSLATION_PENALTY": 3.0,
        },
        # ★★ 新追加：GPU高性能モード
        "GPU高性能": {
            "CONTACT_THRESHOLD": 0.015,
            "TIGHT_THRESHOLD": 0.003,
            "BALANCE_AP_WEIGHT": 0.4,
            "BALANCE_LR_WEIGHT": 0.4,
            "PENETRATION_PENALTY": 0.7,
            "ROTATION_PENALTY": 2.5,
            "TRANSLATION_PENALTY": 3.5,
        },
        # ★★★ 新追加：咬合改善特化モード
        "咬合改善特化": {
            "CONTACT_THRESHOLD": 0.035,      # より広範囲の接触を検出
            "TIGHT_THRESHOLD": 0.012,        # 緩い緊密接触判定
            "BALANCE_AP_WEIGHT": 0.8,        # 前後バランス最重視
            "BALANCE_LR_WEIGHT": 0.8,        # 左右バランス最重視
            "PENETRATION_PENALTY": 0.2,      # めり込み許容
            "ROTATION_PENALTY": 1.0,         # 回転を柔軟に
            "TRANSLATION_PENALTY": 1.5,      # 移動を柔軟に
        },
        # ★★★★ 新追加：均等咬合最適化モード
        "均等咬合最適化": {
            "CONTACT_THRESHOLD": 0.04,       # さらに広範囲の接触検出
            "TIGHT_THRESHOLD": 0.015,        # 均等分布のための緩い判定
            "BALANCE_AP_WEIGHT": 1.2,        # 前後バランス超重視
            "BALANCE_LR_WEIGHT": 1.2,        # 左右バランス超重視
            "PENETRATION_PENALTY": 0.1,      # めり込みを最小限に抑制
            "ROTATION_PENALTY": 0.8,         # 回転をより柔軟に
            "TRANSLATION_PENALTY": 1.2,      # 移動をより柔軟に
        },
        # ★★★★★ 新追加：精密均等咬合モード（沈み込み対策）
        "精密均等咬合": {
            "CONTACT_THRESHOLD": 0.035,      # 適度な接触範囲
            "TIGHT_THRESHOLD": 0.01,         # 沈み込み厳密検出
            "BALANCE_AP_WEIGHT": 1.5,        # 前後バランス最重視
            "BALANCE_LR_WEIGHT": 1.8,        # 左右バランス特に重視（右側改善）
            "PENETRATION_PENALTY": 0.8,      # 沈み込み強力抑制
            "ROTATION_PENALTY": 1.2,         # 適度な回転制限
            "TRANSLATION_PENALTY": 1.8,      # 移動制限強化（沈み込み防止）
        },
    }
    
    def __init__(self, preset="バランス型"):
        """プリセットから設定を初期化"""
        if preset not in self.PRESETS:
            print(f"警告: 不明なプリセット '{preset}' です。デフォルトの 'バランス型' を使用します。")
            preset = "バランス型"
            
        config = self.PRESETS[preset]
        for key, value in config.items():
            setattr(self, key, value)
        
        # 共通設定
        self.MAX_ROTATION = 5.0
        self.MAX_TRANSLATION = 0.6
        self.CLOSE_STEP = -0.05
        
        # GPU設定（安全のためデフォルトはCPU）
        self.USE_GPU = False  # 初期はCPUモード、手動でGPU選択可能
        

        
        # プリセットごとに“重さ”を変える
        if preset == "高速テスト用":
            # 超高速モード：最小限の処理で動作確認
            self.SAMPLE_SIZE = 300           # 2000 → 300（さらに削減）
            self.MAX_CLOSE_STEPS = 10        # 40 → 10
            self.NUM_MULTISTART = 1          # マルチスタート無し（1本のみ）
            self.MAX_LBFGS_ITER = 30         # 100 → 30（さらに削減）
        elif preset == "GPU高性能":
            # GPU高性能モード：GPUの並列性を最大活用
            self.SAMPLE_SIZE = 5000          # より多くのサンプル
            self.MAX_CLOSE_STEPS = 60        # より詳細な接近
            self.NUM_MULTISTART = 8          # より多くの候補
            self.MAX_LBFGS_ITER = 150        # より詳細な最適化
        elif preset == "咬合改善特化":
            # 咬合改善特化モード：接触点の偏りを改善
            self.SAMPLE_SIZE = 2500          # 高精度サンプリング
            self.MAX_CLOSE_STEPS = 40        # 詳細な接触確立
            self.NUM_MULTISTART = 5          # 複数候補で最適解探索
            self.MAX_LBFGS_ITER = 120        # 詳細最適化
        elif preset == "均等咬合最適化":
            # 均等咬合最適化モード：最も均等な咬合分布を追求
            self.SAMPLE_SIZE = 3000          # 最高精度サンプリング
            self.MAX_CLOSE_STEPS = 50        # 最も詳細な接触確立
            self.NUM_MULTISTART = 7          # 豊富な候補から最適解選択
            self.MAX_LBFGS_ITER = 100        # バランス重視の最適化
        elif preset == "精密均等咬合":
            # 精密均等咬合モード：沈み込み対策と右側改善
            self.SAMPLE_SIZE = 2800          # 高精度だが処理時間考慮
            self.MAX_CLOSE_STEPS = 35        # 段階的接近（沈み込み防止）
            self.NUM_MULTISTART = 9          # より多くの候補（最適解探索）
            self.MAX_LBFGS_ITER = 90         # 精密最適化
        else:
            # 高精度モード：接触分析の精度を向上
            self.SAMPLE_SIZE = 1500          # 精度向上（2000→1500で安全性確保）
            self.MAX_CLOSE_STEPS = 30        # 接触確立の詳細化
            self.NUM_MULTISTART = 3          # 複数候補生成
            self.MAX_LBFGS_ITER = 80         # 最適化精度向上
        
        self.preset_name = preset
        
        # 設定値の検証
        self._validate_config()
    
    def _validate_config(self):
        """設定値の妥当性を検証"""
        if self.CONTACT_THRESHOLD <= 0:
            raise ValueError("CONTACT_THRESHOLD は正の値である必要があります")
        if self.TIGHT_THRESHOLD <= 0 or self.TIGHT_THRESHOLD >= self.CONTACT_THRESHOLD:
            raise ValueError("TIGHT_THRESHOLD は 0 < TIGHT_THRESHOLD < CONTACT_THRESHOLD である必要があります")
        if self.SAMPLE_SIZE <= 0:
            raise ValueError("SAMPLE_SIZE は正の整数である必要があります")
        if self.NUM_MULTISTART <= 0:
            raise ValueError("NUM_MULTISTART は正の整数である必要があります")
        
        # 計算時間の警告
        estimated_time = (self.SAMPLE_SIZE * self.NUM_MULTISTART * self.MAX_LBFGS_ITER) / 100000
        if estimated_time > 300:  # 5分以上
            print(f"⚠ 警告: 推定処理時間が {estimated_time/60:.1f} 分を超える可能性があります")


class ContactAnalyzer:
    """接触分析の詳細クラス"""
    
    def __init__(self, sample_vertices, sample_areas, upper, x_mid, y_mid, config):
        self.sample_vertices = sample_vertices
        self.sample_areas = sample_areas
        self.upper = upper
        self.x_mid = x_mid
        self.y_mid = y_mid
        self.config = config
        self.use_gpu = getattr(config, 'USE_GPU', False)
        
        # GPU用データの事前転送
        if self.use_gpu and GPU_AVAILABLE:
            try:
                # GPU初期化テスト
                test_array = cp.array([1, 2, 3])
                _ = cp.linalg.norm(test_array)  # cuBLAS 動作確認
                
                self.sample_vertices_gpu = cp.asarray(sample_vertices)
                self.sample_areas_gpu = cp.asarray(sample_areas)
                self.upper_vertices_gpu = cp.asarray(upper.vertices)
                print("✅ GPU用データを転送しました")
            except Exception as e:
                print(f"⚠ GPU初期化失敗、CPUモードに切り替え: {e}")
                print("   → CUDA Toolkit のインストールが必要な可能性があります")
                self.use_gpu = False
                config.USE_GPU = False  # 設定も更新
    
    def analyze(self, tx, ty, rx, ry, tz):
        """詳細な接触分析を実行（GPU対応）"""
        try:
            rot = R.from_euler("xyz", [rx, ry, 0.0]).as_matrix()
            
            # GPU使用の場合はGPU計算を優先
            if hasattr(self, 'use_gpu') and self.use_gpu and GPU_AVAILABLE:
                try:
                    transformed = self._gpu_transform_vertices(rot, tx, ty, tz)
                    distances = self._gpu_distance_calculation(transformed)
                    # GPU計算成功時はCPUに戻して従来の処理を続行
                    transformed = cp.asnumpy(transformed)
                    closest_points, _, triangle_id = self.upper.nearest.on_surface(transformed)
                except Exception as e:
                    # GPU計算失敗時はCPU計算にフォールバック（エラー出力は制限）
                    if not hasattr(self, '_gpu_error_logged'):
                        print(f"⚠ GPU計算エラー、CPUにフォールバック: {str(e)[:50]}...")
                        self._gpu_error_logged = True
                        self.use_gpu = False  # 以降はCPUを使用
                    
                    transformed = np.dot(self.sample_vertices, rot.T) + np.array([tx, ty, tz])
                    closest_points, distances, triangle_id = self.upper.nearest.on_surface(transformed)
            else:
                # CPU計算
                transformed = np.dot(self.sample_vertices, rot.T) + np.array([tx, ty, tz])
                closest_points, distances, triangle_id = self.upper.nearest.on_surface(transformed)
                
        except Exception as e:
            print(f"警告: 接触分析でエラーが発生しました: {e}")
            return self._create_empty_analysis()
        
        # 接触点の分類
        contact_mask = distances <= self.config.CONTACT_THRESHOLD
        tight_mask = distances <= self.config.TIGHT_THRESHOLD
        
        contact_idx = np.where(contact_mask)[0]
        tight_idx = np.where(tight_mask)[0]
        
        # 詳細な領域分析
        analysis = {
            'total_area': 0.0,
            'anterior_area': 0.0,
            'posterior_area': 0.0,
            'left_area': 0.0,
            'right_area': 0.0,
            'tight_area': 0.0,
            'contact_points': [],
            'distances': distances,
            'transformed_vertices': transformed,
            'contact_mask': contact_mask,
            'num_contact_points': len(contact_idx)
        }
        
        if len(contact_idx) > 0:
            c_areas = self.sample_areas[contact_idx]
            c_pts = self.sample_vertices[contact_idx]
            c_transformed = transformed[contact_idx]
            
            x = c_pts[:, 0]
            y = c_pts[:, 1]
            
            analysis['total_area'] = float(c_areas.sum())
            
            # 前後分析
            anterior_mask = y >= self.y_mid
            posterior_mask = y < self.y_mid
            analysis['anterior_area'] = float(c_areas[anterior_mask].sum())
            analysis['posterior_area'] = float(c_areas[posterior_mask].sum())
            
            # 左右分析
            left_mask = x <= self.x_mid
            right_mask = x > self.x_mid
            analysis['left_area'] = float(c_areas[left_mask].sum())
            analysis['right_area'] = float(c_areas[right_mask].sum())
            
            # 接触点の情報
            analysis['contact_points'] = c_transformed.tolist()
        
        if len(tight_idx) > 0:
            analysis['tight_area'] = float(self.sample_areas[tight_idx].sum())
        
        # バランススコア計算
        analysis['ap_balance'] = min(analysis['anterior_area'], analysis['posterior_area'])
        analysis['lr_balance'] = min(analysis['left_area'], analysis['right_area'])
        
        # 総合スコア（沈み込み対策強化）
        rot_penalty = abs(rx) + abs(ry)
        trans_penalty = np.linalg.norm([tx, ty, tz])
        
        # 沈み込み深度ペナルティ（tz が負の値が大きいほど強いペナルティ）
        depth_penalty = max(0, -tz * 2.0) if tz < 0 else 0
        
        # 右側咬合不足ペナルティ
        lr_imbalance = abs(analysis['left_area'] - analysis['right_area'])
        
        analysis['score'] = (
            analysis['total_area']
            + self.config.BALANCE_AP_WEIGHT * analysis['ap_balance']
            + self.config.BALANCE_LR_WEIGHT * analysis['lr_balance']
            - self.config.PENETRATION_PENALTY * analysis['tight_area']
            - self.config.ROTATION_PENALTY * rot_penalty
            - self.config.TRANSLATION_PENALTY * trans_penalty
            - depth_penalty  # 沈み込み深度ペナルティ追加
            - 0.3 * lr_imbalance  # 左右不均衡ペナルティ追加
        )
        
        return analysis
    
    def _create_empty_analysis(self):
        """エラー時の空の分析結果を作成"""
        return {
            'total_area': 0.0,
            'anterior_area': 0.0,
            'posterior_area': 0.0,
            'left_area': 0.0,
            'right_area': 0.0,
            'tight_area': 0.0,
            'contact_points': [],
            'distances': np.array([]),
            'transformed_vertices': np.array([]),
            'contact_mask': np.array([]),
            'num_contact_points': 0,
            'ap_balance': 0.0,
            'lr_balance': 0.0,
            'score': -1000.0  # 最低スコア
        }
    
    def _gpu_transform_vertices(self, rot, tx, ty, tz):
        """GPU上で頂点変換を実行"""
        rot_gpu = cp.asarray(rot)
        translation_gpu = cp.asarray([tx, ty, tz])
        return cp.dot(self.sample_vertices_gpu, rot_gpu.T) + translation_gpu
    
    def _gpu_distance_calculation(self, transformed_gpu):
        """GPU上で距離計算を実行"""
        # 各サンプル点から上顎の全頂点までの距離を計算
        # (N, 1, 3) - (1, M, 3) -> (N, M, 3) -> (N, M) -> (N,)
        diff = transformed_gpu[:, None, :] - self.upper_vertices_gpu[None, :, :]
        distances_all = cp.linalg.norm(diff, axis=2)
        distances = cp.min(distances_all, axis=1)
        return cp.asnumpy(distances)  # CPUに戻す


def select_two_stl_files():
    """STLファイルを2つ選択"""
    root = Tk()
    root.withdraw()

    filepaths = filedialog.askopenfilenames(
        title="上顎と下顎の STL ファイルをこの順に2つ選択してください（1つ目: 上顎, 2つ目: 下顎）",
        filetypes=[("STL files", "*.stl"), ("All files", "*.*")]
    )
    root.update()
    root.destroy()

    if len(filepaths) != 2:
        print("エラー: STL ファイルは必ず 2 つ選択してください。")
        sys.exit(1)

    upper_path, lower_path = filepaths
    print("上顎 STL:", upper_path)
    print("下顎 STL:", lower_path)
    return upper_path, lower_path


def load_mesh_safely(filepath):
    """安全なメッシュ読み込み"""
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"ファイルが見つかりません: {filepath}")
        
        mesh = trimesh.load(filepath)
        
        # メッシュの基本検証
        if mesh is None:
            raise ValueError("メッシュの読み込みに失敗しました")
        
        if hasattr(mesh, '__len__') and len(mesh) > 1:
            # 複数のメッシュが含まれている場合、最大のものを選択
            mesh = max(mesh, key=lambda m: len(m.vertices) if hasattr(m, 'vertices') else 0)
            print(f"複数メッシュが検出されました。最大メッシュを選択しています。")
        
        if not hasattr(mesh, 'vertices') or not hasattr(mesh, 'faces'):
            raise ValueError("無効なメッシュ形式です")
        
        if len(mesh.vertices) < 100:
            raise ValueError(f"頂点数が少なすぎます: {len(mesh.vertices)}")
        
        if len(mesh.faces) < 50:
            raise ValueError(f"面数が少なすぎます: {len(mesh.faces)}")
        
        if not mesh.is_watertight:
            print(f"警告: {os.path.basename(filepath)} は水密ではありません")
            # 可能であれば修復を試行
            try:
                mesh.fill_holes()
                if mesh.is_watertight:
                    print("  → 穴の修復に成功しました")
            except:
                pass
        
        print(f"✓ {os.path.basename(filepath)} 読み込み成功 ({len(mesh.vertices)} 頂点, {len(mesh.faces)} 面)")
        return mesh
        
    except Exception as e:
        print(f"エラー: {filepath} の読み込みに失敗しました")
        print(f"詳細: {e}")
        print("対処方法:")
        print("  1. ファイルパスが正しいか確認してください")
        print("  2. STLファイルが破損していないか確認してください") 
        print("  3. ファイルの読み取り権限があるか確認してください")
        sys.exit(1)


def per_vertex_area(mesh: trimesh.Trimesh):
    """頂点ごとの代表面積を計算"""
    areas = np.zeros(len(mesh.vertices))
    for face, area in zip(mesh.faces, mesh.area_faces):
        for vid in face:
            areas[vid] += area / 3.0
    return areas


def gpu_accelerated_distance_calc(vertices, target_mesh, use_gpu=False):
    """GPU加速された距離計算（オプション）"""
    if use_gpu and GPU_AVAILABLE:
        try:
            # CupyでGPU計算
            vertices_gpu = cp.asarray(vertices)
            target_vertices_gpu = cp.asarray(target_mesh.vertices)
            
            # メモリ効率的な距離計算（チャンク処理）
            chunk_size = 1000  # メモリ使用量を制御
            distances = []
            
            for i in range(0, len(vertices), chunk_size):
                chunk_vertices = vertices_gpu[i:i+chunk_size]
                # ブロードキャストで距離計算
                diff = chunk_vertices[:, None, :] - target_vertices_gpu[None, :, :]
                chunk_distances = cp.linalg.norm(diff, axis=2).min(axis=1)
                distances.append(chunk_distances)
            
            # 結果をCPUに戻す
            return cp.asnumpy(cp.concatenate(distances))
        except Exception as e:
            print(f"GPU計算でエラー、CPUにフォールバック: {e}")
            return target_mesh.nearest.on_surface(vertices)[1]
    else:
        # 従来のCPU計算
        return target_mesh.nearest.on_surface(vertices)[1]


def gpu_batch_optimization(analyzer, param_candidates, config):
    """GPU上で複数パラメータを並列最適化"""
    if not (GPU_AVAILABLE and getattr(config, 'USE_GPU', False)):
        return None
    
    try:
        # 複数のパラメータセットを同時にGPUで評価
        batch_scores = []
        param_tensor = cp.asarray(param_candidates)
        
        # バッチ処理でスコア計算
        for params in param_tensor:
            tx, ty, rx, ry, tz = params
            analysis = analyzer.analyze(tx, ty, rx, ry, tz)
            batch_scores.append(analysis['score'])
        
        return cp.asnumpy(cp.asarray(batch_scores))
    except Exception as e:
        print(f"GPU バッチ最適化エラー: {e}")
        return None


def enable_gpu_optimization():
    """GPU最適化の有効化"""
    global GPU_AVAILABLE
    return GPU_AVAILABLE and input("🚀 GPU加速を使用しますか？ (y/n): ").lower() == 'y'


def close_until_first_contact(analyzer, config):
    """初期接触位置を見つける（タイムアウト付き）"""
    import time
    import signal
    
    start_time = time.time()
    timeout = 300  # 5分でタイムアウト
    
    tx, ty = 0.0, 0.0
    rx, ry = 0.0, 0.0
    tz = 0.0
    
    print("\n[ステージ1: 初期接触の確立]")
    print(f"  最大時間: {timeout//60}分でタイムアウトします")
    
    last_analysis = None
    
    for i in range(min(config.MAX_CLOSE_STEPS, 20)):  # 最大20ステップに制限
        # タイムアウトチェック
        if time.time() - start_time > timeout:
            print("⚠ タイムアウト: 処理を強制停止します")
            break
            
        tz_new = tz + config.CLOSE_STEP
        
        try:
            step_start = time.time()
            analysis = analyzer.analyze(tx, ty, rx, ry, tz_new)
            step_time = time.time() - step_start
            
            # 1ステップが30秒以上の場合は異常
            if step_time > 30:
                print(f"⚠ 処理時間異常 ({step_time:.1f}秒), 処理を停止します")
                break
                
        except KeyboardInterrupt:
            print("\n⚠ ユーザーによる停止要求")
            break
        except Exception as e:
            print(f"⚠ 処理エラー: {e}")
            analysis = analyzer._create_empty_analysis()
        
        if (i + 1) % 2 == 0:  # より頻繁に進捗表示
            elapsed = time.time() - start_time
            print(f"  ステップ {i+1}/{min(config.MAX_CLOSE_STEPS, 20)}: tz={tz_new:.3f}mm, 接触面積={analysis['total_area']:.4f} (経過:{elapsed:.1f}秒)")
        
        tz = tz_new
        last_analysis = analysis
        
        if analysis['total_area'] > 0.0:
            print(f"✓ 初期接触確立: tz={tz:.3f}mm, 接触面積={analysis['total_area']:.4f}mm²")
            return tx, ty, rx, ry, tz, analysis
    
    print("⚠ 警告: 接触が見つかりませんでした（最後の状態を返します）")
    return tx, ty, rx, ry, tz, last_analysis or analyzer._create_empty_analysis()


def optimize_single_position(analyzer, initial_params, config):
    """単一の初期位置から最適化（タイムアウト付き）"""
    import time
    
    start_time = time.time()
    timeout = 120  # 2分でタイムアウト
    
    iteration_count = [0]  # クロージャで使うためリストに
    best_score = [-float('inf')]  # 最良スコアを記録
    
    def objective(params):
        # タイムアウトチェック（一度だけ表示）
        if time.time() - start_time > timeout:
            if not hasattr(objective, '_timeout_shown'):
                print(f"\n    ⚠ 最適化タイムアウト ({timeout//60}分) - 処理を終了します")
                objective._timeout_shown = True
            return float('inf')  # 最適化を強制終了
            
        tx, ty, rx, ry, tz = params
        
        try:
            analysis = analyzer.analyze(tx, ty, rx, ry, tz)
        except Exception as e:
            print(f"\n    ⚠ 解析エラー: {str(e)[:30]}...")
            return float('inf')
        
        # 進捗表示とベストスコア更新
        iteration_count[0] += 1
        if analysis['score'] > best_score[0]:
            best_score[0] = analysis['score']
        
        if iteration_count[0] % 10 == 0:  # より頻繁に表示
            elapsed = time.time() - start_time
            progress = min(100, (iteration_count[0] / min(config.MAX_LBFGS_ITER, 50)) * 100)
            print(f"    進捗: {progress:.0f}% | ベスト: {best_score[0]:.3f} | 経過: {elapsed:.0f}秒", end='\r')
        
        return -analysis['score']
    
    bounds = [
        (-config.MAX_TRANSLATION, config.MAX_TRANSLATION),
        (-config.MAX_TRANSLATION, config.MAX_TRANSLATION),
        (-np.deg2rad(config.MAX_ROTATION), np.deg2rad(config.MAX_ROTATION)),
        (-np.deg2rad(config.MAX_ROTATION), np.deg2rad(config.MAX_ROTATION)),
        (-1.5, 1.0)
    ]
    
    result = minimize(
        objective,
        initial_params,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': config.MAX_LBFGS_ITER, 'disp': False}
    )
    
    print()  # 改行
    return result


def multistart_optimization(analyzer, base_params, config):
    """マルチスタート最適化で複数の候補を生成（GPU並列対応）"""
    
    gpu_status = "🚀 GPU" if getattr(config, 'USE_GPU', False) and GPU_AVAILABLE else "💻 CPU"
    print(f"\n[ステージ2: マルチスタート最適化] ({gpu_status}モード)")
    print(f"  {config.NUM_MULTISTART}個の初期位置から最適化を実行...\n")
    
    candidates = []
    
    # ベース位置からの最適化
    print(f"  [1/{config.NUM_MULTISTART}] ベース位置から最適化中...")
    result = optimize_single_position(analyzer, base_params, config)
    tx, ty, rx, ry, tz = result.x
    analysis = analyzer.analyze(tx, ty, rx, ry, tz)
    print(f"    ✓ 完了: スコア={analysis['score']:.4f}")
    candidates.append({
        'params': (tx, ty, rx, ry, tz),
        'analysis': analysis,
        'source': 'ベース位置'
    })
    
    # 摂動を加えた初期位置から最適化（高速テスト用では NUM_MULTISTART=1 なのでここはスキップされる）
    rng = np.random.default_rng(42)
    for i in range(config.NUM_MULTISTART - 1):
        perturbed = np.array(base_params).copy()
        perturbed[0] += rng.uniform(-0.2, 0.2)
        perturbed[1] += rng.uniform(-0.2, 0.2)
        perturbed[2] += rng.uniform(-np.deg2rad(2), np.deg2rad(2))
        perturbed[3] += rng.uniform(-np.deg2rad(2), np.deg2rad(2))
        perturbed[4] += rng.uniform(-0.1, 0.1)
        
        print(f"\n  [{i+2}/{config.NUM_MULTISTART}] 摂動位置 {i+1} から最適化中...")
        result = optimize_single_position(analyzer, perturbed, config)
        tx, ty, rx, ry, tz = result.x
        analysis = analyzer.analyze(tx, ty, rx, ry, tz)
        print(f"    ✓ 完了: スコア={analysis['score']:.4f}")
        candidates.append({
            'params': (tx, ty, rx, ry, tz),
            'analysis': analysis,
            'source': f'摂動位置 {i+1}'
        })
    
    # スコアでソート（均等咬合最適化の場合はバランス重視）
    if hasattr(config, 'preset_name') and config.preset_name in ["均等咬合最適化", "精密均等咬合"]:
        # バランススコア重視でソート
        def balance_score(cand):
            analysis = cand['analysis']
            base_score = analysis['score']
            # 前後・左右バランスが均等であるほど高スコア
            ap_balance_ratio = min(analysis['anterior_area'], analysis['posterior_area']) / (max(analysis['anterior_area'], analysis['posterior_area']) + 1e-6)
            lr_balance_ratio = min(analysis['left_area'], analysis['right_area']) / (max(analysis['left_area'], analysis['right_area']) + 1e-6)
            balance_bonus = (ap_balance_ratio + lr_balance_ratio) * 0.5
            
            # 沈み込み深度ペナルティ（tz値を考慮）
            tx, ty, rx, ry, tz = cand['params']
            depth_penalty = max(0, -tz * 3.0) if tz < -0.05 else 0
            
            return base_score + balance_bonus - depth_penalty
        candidates.sort(key=balance_score, reverse=True)
    else:
        candidates.sort(key=lambda x: x['analysis']['score'], reverse=True)
    
    print(f"\n✓ {len(candidates)}個の候補位置を生成")
    for i, cand in enumerate(candidates[:3]):
        print(f"  候補{i+1}: スコア={cand['analysis']['score']:.4f}, "
              f"接触面積={cand['analysis']['total_area']:.2f}mm²")
    
    return candidates


def generate_interactive_html(candidates, upper, lower_refined_list, output_dir, config):
    """インタラクティブなHTMLレポートを生成"""
    
    print("\nインタラクティブなHTMLレポートを生成中...")
    
    html_content = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>咬頭嵌合位最適化レポート</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            padding: 30px;
        }}
        h1 {{
            color: #667eea;
            text-align: center;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }}
        .config-info {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 30px;
            border-left: 4px solid #667eea;
        }}
        .candidate {{
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 30px;
            transition: all 0.3s;
        }}
        .candidate:hover {{
            border-color: #667eea;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        }}
        .candidate.best {{
            border-color: #28a745;
            background: #f0fff4;
        }}
        .candidate-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        .candidate-title {{
            font-size: 1.3em;
            font-weight: bold;
            color: #667eea;
        }}
        .candidate.best .candidate-title {{
            color: #28a745;
        }}
        .badge {{
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
        }}
        .badge-best {{
            background: #28a745;
            color: white;
        }}
        .badge-rank {{
            background: #6c757d;
            color: white;
        }}
        .params-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .param-box {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .param-label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }}
        .param-value {{
            font-size: 1.3em;
            font-weight: bold;
            color: #333;
        }}
        .analysis-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .analysis-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .analysis-label {{
            font-size: 0.85em;
            opacity: 0.9;
            margin-bottom: 5px;
        }}
        .analysis-value {{
            font-size: 1.4em;
            font-weight: bold;
        }}
        .progress-bar {{
            width: 100%;
            height: 10px;
            background: #e0e0e0;
            border-radius: 5px;
            overflow: hidden;
            margin-top: 5px;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #28a745, #20c997);
            transition: width 0.3s;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }}
        th {{
            background: #f8f9fa;
            font-weight: bold;
            color: #667eea;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🦷 咬頭嵌合位最適化レポート</h1>
        <div class="subtitle">生成日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</div>
        
        <div class="config-info">
            <strong>📋 使用プリセット:</strong> {config.preset_name}<br>
            <strong>🔍 サンプル数:</strong> {config.SAMPLE_SIZE} 頂点<br>
            <strong>🎯 接触閾値:</strong> {config.CONTACT_THRESHOLD} mm<br>
            <strong>🔄 マルチスタート:</strong> {config.NUM_MULTISTART} 回
        </div>
"""
    
    for i, cand in enumerate(candidates):
        tx, ty, rx, ry, tz = cand['params']
        analysis = cand['analysis']
        
        is_best = (i == 0)
        best_class = " best" if is_best else ""
        badge = '<span class="badge badge-best">🏆 最優秀候補</span>' if is_best else f'<span class="badge badge-rank">候補 #{i+1}</span>'
        
        ap_balance_pct = 0
        lr_balance_pct = 0
        if analysis['total_area'] > 0:
            ap_balance_pct = (min(analysis['anterior_area'], analysis['posterior_area']) / 
                             (max(analysis['anterior_area'], analysis['posterior_area']) + 1e-6)) * 100
            lr_balance_pct = (min(analysis['left_area'], analysis['right_area']) / 
                             (max(analysis['left_area'], analysis['right_area']) + 1e-6)) * 100
        
        html_content += f"""
        <div class="candidate{best_class}">
            <div class="candidate-header">
                <div class="candidate-title">候補 {i+1} - {cand['source']}</div>
                {badge}
            </div>
            
            <div class="params-grid">
                <div class="param-box">
                    <div class="param-label">水平移動 X</div>
                    <div class="param-value">{tx:.3f} mm</div>
                </div>
                <div class="param-box">
                    <div class="param-label">水平移動 Y</div>
                    <div class="param-value">{ty:.3f} mm</div>
                </div>
                <div class="param-box">
                    <div class="param-label">垂直移動 Z</div>
                    <div class="param-value">{tz:.3f} mm</div>
                </div>
                <div class="param-box">
                    <div class="param-label">回転 X軸</div>
                    <div class="param-value">{np.rad2deg(rx):.2f}°</div>
                </div>
                <div class="param-box">
                    <div class="param-label">回転 Y軸</div>
                    <div class="param-value">{np.rad2deg(ry):.2f}°</div>
                </div>
            </div>
            
            <div class="analysis-grid">
                <div class="analysis-box">
                    <div class="analysis-label">総合スコア</div>
                    <div class="analysis-value">{analysis['score']:.2f}</div>
                </div>
                <div class="analysis-box">
                    <div class="analysis-label">総接触面積</div>
                    <div class="analysis-value">{analysis['total_area']:.2f} mm²</div>
                </div>
                <div class="analysis-box">
                    <div class="analysis-label">接触点数</div>
                    <div class="analysis-value">{analysis['num_contact_points']}</div>
                </div>
            </div>
            
            <table>
                <tr>
                    <th>領域</th>
                    <th>接触面積 (mm²)</th>
                    <th>比率</th>
                </tr>
                <tr>
                    <td>前歯部</td>
                    <td>{analysis['anterior_area']:.2f}</td>
                    <td>{(analysis['anterior_area']/analysis['total_area']*100 if analysis['total_area']>0 else 0):.1f}%</td>
                </tr>
                <tr>
                    <td>臼歯部</td>
                    <td>{analysis['posterior_area']:.2f}</td>
                    <td>{(analysis['posterior_area']/analysis['total_area']*100 if analysis['total_area']>0 else 0):.1f}%</td>
                </tr>
                <tr>
                    <td>左側</td>
                    <td>{analysis['left_area']:.2f}</td>
                    <td>{(analysis['left_area']/analysis['total_area']*100 if analysis['total_area']>0 else 0):.1f}%</td>
                </tr>
                <tr>
                    <td>右側</td>
                    <td>{analysis['right_area']:.2f}</td>
                    <td>{(analysis['right_area']/analysis['total_area']*100 if analysis['total_area']>0 else 0):.1f}%</td>
                </tr>
            </table>
            
            <div style="margin-top: 20px;">
                <strong>バランス評価:</strong>
                <div style="margin: 10px 0;">
                    前後バランス: {ap_balance_pct:.1f}%
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {ap_balance_pct}%"></div>
                    </div>
                </div>
                <div style="margin: 10px 0;">
                    左右バランス: {lr_balance_pct:.1f}%
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {lr_balance_pct}%"></div>
                    </div>
                </div>
            </div>
        </div>
"""
    
    html_content += """
        <div class="footer">
            <p>💡 このレポートは自動生成されています。臨床判断は歯科医師が行ってください。</p>
            <p>Shibuya Dental Laboratory | Digital Occlusion Analysis System v2.0 (fast)</p>
        </div>
    </div>
</body>
</html>
"""
    
    html_path = os.path.join(output_dir, "optimization_report_interactive.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ インタラクティブHTMLレポート: {html_path}")
    return html_path


def visualize_candidates(candidates, lower, upper, output_dir):
    """複数候補の可視化"""
    
    print("\n候補結果を可視化中...")
    
    num_candidates = min(3, len(candidates))
    fig = plt.figure(figsize=(20, 6 * num_candidates))
    
    for idx, cand in enumerate(candidates[:num_candidates]):
        tx, ty, rx, ry, tz = cand['params']
        analysis = cand['analysis']
        
        rot = R.from_euler("xyz", [rx, ry, 0.0]).as_matrix()
        transformed_vertices = (rot @ lower.vertices.T).T + np.array([tx, ty, tz])
        
        ax1 = fig.add_subplot(num_candidates, 3, idx*3 + 1, projection='3d')
        ax1.plot_trisurf(
            transformed_vertices[:, 0],
            transformed_vertices[:, 1],
            transformed_vertices[:, 2],
            triangles=lower.faces,
            color='lightblue',
            alpha=0.7,
            edgecolor='none'
        )
        ax1.set_title(f'候補{idx+1}: 下顎\nスコア: {analysis["score"]:.2f}', 
                     fontsize=12, fontweight='bold')
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_zlabel('Z (mm)')
        
        ax2 = fig.add_subplot(num_candidates, 3, idx*3 + 2, projection='3d')
        ax2.plot_trisurf(
            upper.vertices[:, 0],
            upper.vertices[:, 1],
            upper.vertices[:, 2],
            triangles=upper.faces,
            color='lightcoral',
            alpha=0.7,
            edgecolor='none'
        )
        ax2.set_title(f'候補{idx+1}: 上顎（固定）', fontsize=12, fontweight='bold')
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        ax2.set_zlabel('Z (mm)')
        
        ax3 = fig.add_subplot(num_candidates, 3, idx*3 + 3)
        distances = analysis['distances']
        ax3.hist(distances[distances < 0.1], bins=50, edgecolor='black', alpha=0.7)
        ax3.axvline(x=0.02, color='r', linestyle='--', linewidth=2, 
                   label=f'Contact: {analysis["num_contact_points"]} points')
        ax3.set_xlabel('Distance (mm)')
        ax3.set_ylabel('Frequency')
        ax3.set_title(f'候補{idx+1}: 接触距離分布', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    viz_path = os.path.join(output_dir, 'candidates_comparison.png')
    plt.savefig(viz_path, dpi=200, bbox_inches='tight')
    print(f"✓ 候補比較画像: {viz_path}")
    plt.close()


def generate_detailed_report(candidates, config, output_dir):
    """詳細テキストレポート生成"""
    
    report = f"""
{'='*80}
咬頭嵌合位最適化 詳細レポート
{'='*80}
生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
使用プリセット: {config.preset_name}

【最適化設定】
  接触閾値:       {config.CONTACT_THRESHOLD} mm
  緊密閾値:       {config.TIGHT_THRESHOLD} mm
  サンプル数:     {config.SAMPLE_SIZE} 頂点
  最大回転制限:   ±{config.MAX_ROTATION}°
  最大移動制限:   ±{config.MAX_TRANSLATION} mm
  マルチスタート: {config.NUM_MULTISTART} 回

【スコア重み設定】
  前後バランス:   {config.BALANCE_AP_WEIGHT}
  左右バランス:   {config.BALANCE_LR_WEIGHT}
  めり込みペナルティ: {config.PENETRATION_PENALTY}
  回転ペナルティ: {config.ROTATION_PENALTY}
  移動ペナルティ: {config.TRANSLATION_PENALTY}

{'='*80}
最適化結果 - 候補一覧
{'='*80}
"""
    
    for i, cand in enumerate(candidates):
        tx, ty, rx, ry, tz = cand['params']
        analysis = cand['analysis']
        
        ap_balance_pct = 0
        lr_balance_pct = 0
        if analysis['total_area'] > 0:
            ap_balance_pct = (min(analysis['anterior_area'], analysis['posterior_area']) / 
                             (max(analysis['anterior_area'], analysis['posterior_area']) + 1e-6)) * 100
            lr_balance_pct = (min(analysis['left_area'], analysis['right_area']) / 
                             (max(analysis['left_area'], analysis['right_area']) + 1e-6)) * 100
        
        best_marker = " ★ 最優秀候補" if i == 0 else ""
        
        report += f"""
【候補 {i+1}{best_marker}】
出所: {cand['source']}

■ 位置パラメータ:
  水平移動 X軸:   {tx:8.3f} mm
  水平移動 Y軸:   {ty:8.3f} mm
  垂直移動 Z軸:   {tz:8.3f} mm
  回転 X軸:       {np.rad2deg(rx):8.2f}°
  回転 Y軸:       {np.rad2deg(ry):8.2f}°

■ 接触評価:
  総合スコア:     {analysis['score']:8.2f}
  総接触面積:     {analysis['total_area']:8.2f} mm²
  接触点数:       {analysis['num_contact_points']:8d} points
  
  前歯部接触:     {analysis['anterior_area']:8.2f} mm² ({analysis['anterior_area']/analysis['total_area']*100 if analysis['total_area']>0 else 0:5.1f}%)
  臼歯部接触:     {analysis['posterior_area']:8.2f} mm² ({analysis['posterior_area']/analysis['total_area']*100 if analysis['total_area']>0 else 0:5.1f}%)
  左側接触:       {analysis['left_area']:8.2f} mm² ({analysis['left_area']/analysis['total_area']*100 if analysis['total_area']>0 else 0:5.1f}%)
  右側接触:       {analysis['right_area']:8.2f} mm² ({analysis['right_area']/analysis['total_area']*100 if analysis['total_area']>0 else 0:5.1f}%)
  
  緊密接触面積:   {analysis['tight_area']:8.2f} mm²

■ バランス評価:
  前後バランス率: {ap_balance_pct:8.1f}%
  左右バランス率: {lr_balance_pct:8.1f}%

{'-'*80}
"""
    
    report += f"""
{'='*80}
推奨事項
{'='*80}

最優秀候補（候補1）の使用を推奨します。

ただし、以下の点に注意してください：
1. このシミュレーションは剛体近似に基づいています
2. 生体の弾性変形や顎関節の可動性は考慮されていません
3. 最終的な臨床判断は歯科医師が行ってください
4. 候補2以降も参考として確認することを推奨します

{'='*80}
Shibuya Dental Laboratory
Digital Occlusion Analysis System v2.0 (fast)
{'='*80}
"""
    
    report_path = os.path.join(output_dir, "optimization_report_detailed.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✓ 詳細テキストレポート: {report_path}")
    print("\n" + "="*80)
    print("最優秀候補のサマリー:")
    print("="*80)
    
    best = candidates[0]
    tx, ty, rx, ry, tz = best['params']
    analysis = best['analysis']
    
    print(f"  総合スコア:     {analysis['score']:.2f}")
    print(f"  総接触面積:     {analysis['total_area']:.2f} mm²")
    print(f"  接触点数:       {analysis['num_contact_points']} points")
    print(f"  水平移動:       X={tx:.3f}mm, Y={ty:.3f}mm")
    print(f"  垂直移動:       Z={tz:.3f}mm")
    print(f"  回転:           X={np.rad2deg(rx):.2f}°, Y={np.rad2deg(ry):.2f}°")
    print("="*80)


def main():
    """メイン処理"""
    setup_logging()
    
    print("="*80)
    print("咬頭嵌合位自動最適化システム v2.1 (改良版)")
    print("Advanced Digital Occlusion Analysis")
    print("="*80)
    
    start_time = datetime.now()
    logging.info("最適化処理を開始しました")

    # プリセット選択
    print("\n使用するプリセットを選択してください:")
    presets = list(OptimizationConfig.PRESETS.keys())
    for i, preset in enumerate(presets):
        gpu_mark = " 🚀" if preset == "GPU高性能" and GPU_AVAILABLE else ""
        cpu_mark = " (推奨)" if preset == "高速テスト用" and not GPU_AVAILABLE else ""
        print(f"  {i+1}. {preset}{gpu_mark}{cpu_mark}")
    
    # GPU が利用可能な場合はGPU高性能モードを推奨
    default_choice = 6 if GPU_AVAILABLE else 5  # GPU高性能 or 高速テスト用
    
    while True:
        try:
            choice = input(f"\n選択 (1-{len(presets)}) [デフォルト: {default_choice}]: ").strip()
            if choice == "":
                choice = default_choice
            else:
                choice = int(choice)
            
            if 1 <= choice <= len(presets):
                selected_preset = presets[choice - 1]
                break
            else:
                print("無効な選択です。もう一度入力してください。")
        except ValueError:
            print("数字を入力してください。")
    
    config = OptimizationConfig(preset=selected_preset)
    print(f"\n✓ 選択されたプリセット: {selected_preset}")
    
    # GPU使用の最終確認
    if GPU_AVAILABLE:
        if selected_preset == "GPU高性能":
            config.USE_GPU = True
            print("🚀 GPU高性能モードが自動選択されました")
        elif config.USE_GPU:  # デフォルトでGPU使用に設定されている場合
            use_gpu_choice = input("🚀 GPUを使用しますか？ (Y/n): ").strip().lower()
            config.USE_GPU = use_gpu_choice != 'n'
        
        if config.USE_GPU:
            print("✅ GPU加速モードで実行します")
            print(f"   サンプル数: {config.SAMPLE_SIZE} 頂点（GPU並列処理）")
        else:
            print("💻 CPUモードで実行します")
    else:
        config.USE_GPU = False
        print("💻 CPUモードで実行します")
        print("   ※ GPU を使用したい場合は、CuPy と CUDA Toolkit が正常にインストールされているか確認してください")

    # ここから下は「前と同じ」でOK（ファイル選択〜最適化〜保存の流れ）
    upper_path, lower_path = select_two_stl_files()
    output_dir = os.path.dirname(lower_path)

    print("\nメッシュを読み込み中...")
    upper = load_mesh_safely(upper_path)
    lower = load_mesh_safely(lower_path)

    print("\n頂点面積を計算中...")
    lower_vertex_area_all = per_vertex_area(lower)

    all_vertices = lower.vertices
    n_vertices = len(all_vertices)

    if n_vertices > config.SAMPLE_SIZE:
        rng = np.random.default_rng(0)
        sample_idx = rng.choice(n_vertices, size=config.SAMPLE_SIZE, replace=False)
        print(f"✓ {n_vertices} 頂点から {config.SAMPLE_SIZE} 頂点をサンプリング")
    else:
        sample_idx = np.arange(n_vertices)
        print(f"✓ 全 {n_vertices} 頂点を使用")

    sample_vertices = all_vertices[sample_idx]
    sample_areas = lower_vertex_area_all[sample_idx]

    x_mid = float(np.median(sample_vertices[:, 0]))
    y_mid = float(np.median(sample_vertices[:, 1]))
    print(f"  左右の境界 (x_mid) = {x_mid:.4f} mm")
    print(f"  前後の境界 (y_mid) = {y_mid:.4f} mm")

    analyzer = ContactAnalyzer(sample_vertices, sample_areas, upper, x_mid, y_mid, config)

    tx0, ty0, rx0, ry0, tz0, analysis0 = close_until_first_contact(analyzer, config)

    base_params = [tx0, ty0, rx0, ry0, tz0]
    candidates = multistart_optimization(analyzer, base_params, config)

    print("\n最適化された下顎STLを保存中...")
    lower_refined_list = []
    lower_name = os.path.splitext(os.path.basename(lower_path))[0]

    for i, cand in enumerate(candidates[:3]):
        tx, ty, rx, ry, tz = cand['params']
        rot = R.from_euler("xyz", [rx, ry, 0.0]).as_matrix()
        transformed_vertices = (rot @ lower.vertices.T).T + np.array([tx, ty, tz])

        lower_refined = lower.copy()
        lower_refined.vertices = transformed_vertices
        lower_refined_list.append(lower_refined)

        out_path = os.path.join(output_dir, f"{lower_name}_optimized_candidate{i+1}.stl")
        lower_refined.export(out_path)
        print(f"  候補{i+1}: {out_path}")

    if ENABLE_PLOT:
        visualize_candidates(candidates, lower, upper, output_dir)

    generate_detailed_report(candidates, config, output_dir)
    if ENABLE_HTML:
        generate_interactive_html(candidates, upper, lower_refined_list, output_dir, config)

    # 処理時間統計
    end_time = datetime.now()
    total_time = end_time - start_time
    logging.info(f"処理が完了しました。総処理時間: {total_time}")
    
    print("\n" + "="*80)
    print("🎉 すべての処理が完了しました！")
    print("="*80)
    print(f"\n⏱ 処理時間: {total_time}")
    print(f"📊 使用プリセット: {config.preset_name}")
    print(f"🔍 解析頂点数: {len(sample_vertices)}")
    print(f"🎯 生成候補数: {len(candidates)}")
    
    print("\n生成されたファイル:")
    print(f"  📄 詳細レポート: optimization_report_detailed.txt")
    print(f"  📊 ログファイル: optimization.log")
    if ENABLE_HTML:
        print(f"  🌐 HTMLレポート: optimization_report_interactive.html")
    if ENABLE_PLOT:
        print(f"  📊 候補比較画像: candidates_comparison.png")
    print(f"  🦷 STLファイル: {lower_name}_optimized_candidate1-3.stl")
    
    if config.preset_name == "高速テスト用":
        print("\n※ 高速テスト用プリセットで結果を確認し、良さそうなら他プリセットでも試してみてください。")
    
    print("="*80)



if __name__ == "__main__":
    main()

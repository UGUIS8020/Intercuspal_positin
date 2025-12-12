import os
import sys
import numpy as np
import trimesh
from tkinter import Tk, filedialog
from scipy.spatial.transform import Rotation as R


# =============================
# ユーティリティ
# =============================

def select_two_stl_files():
    """
    ファイルダイアログから STL ファイルを2つ選択
    1つ目: 上顎, 2つ目: 下顎 として扱う
    """
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
# スコアリング（輪ゴム / バネモデル）
# =============================

class SpringOcclusionScorer:
    """
    「輪ゴムでぐるぐる巻いた」イメージで、
    接触点を仮想バネとして扱うスコア計算クラス
    """

    def __init__(
        self,
        upper_mesh: trimesh.Trimesh,
        lower_sample_vertices: np.ndarray,
        lower_sample_areas: np.ndarray,
        contact_threshold: float = 0.03,
        rot_penalty: float = 1.5,
        trans_penalty: float = 2.0,
        balance_ap_weight: float = 0.4,
        balance_lr_weight: float = 0.4,
    ):
        self.upper = upper_mesh
        self.v0 = lower_sample_vertices  # 形態（座標）の基準
        self.areas = lower_sample_areas
        self.contact_threshold = contact_threshold
        self.rot_penalty = rot_penalty
        self.trans_penalty = trans_penalty
        self.balance_ap_weight = balance_ap_weight
        self.balance_lr_weight = balance_lr_weight

        # 左右 / 前後の境界（下顎の元の座標系で決めておく）
        self.x_mid = float(np.median(self.v0[:, 0]))
        self.y_mid = float(np.median(self.v0[:, 1]))
        print(f"  左右の境界 (x_mid) = {self.x_mid:.4f} mm")
        print(f"  前後の境界 (y_mid) = {self.y_mid:.4f} mm")

    def evaluate(self, rx_rad, ry_rad, tz, max_dist_clip=0.1):
        """
        姿勢 (rx, ry, tz) に対するスコアを返す
        - rx, ry: ラジアン（X, Y軸まわりの回転）
        - tz: mm（垂直方向の出し入れ）
        tx, ty は 0 固定（バイト位置を原点とみなす）
        """
        tx, ty = 0.0, 0.0

        # 剛体変換（下顎サンプル頂点）
        rot = R.from_euler("xyz", [rx_rad, ry_rad, 0.0]).as_matrix()
        transformed = (rot @ self.v0.T).T + np.array([tx, ty, tz])

        # 上顎メッシュとの最近接距離（trimesh が内部でkd-treeをキャッシュ）
        closest_points, distances, tri_id = self.upper.nearest.on_surface(transformed)

        # 「輪ゴムが届いている」範囲：contact_threshold 以内
        d = np.clip(distances, 0.0, max_dist_clip)
        contact_mask = d <= self.contact_threshold

        if not np.any(contact_mask):
            # まったく噛んでいないなら、ほぼ0スコア（少しだけ回転・移動ペナルティ）
            penalty = self.rot_penalty * (abs(rx_rad) + abs(ry_rad)) + \
                      self.trans_penalty * abs(tz)
            return -penalty, {
                "total_area": 0.0,
                "anterior_area": 0.0,
                "posterior_area": 0.0,
                "left_area": 0.0,
                "right_area": 0.0,
                "num_contacts": 0,
                "rx": rx_rad,
                "ry": ry_rad,
                "tz": tz,
            }

        contact_d = d[contact_mask]
        contact_a = self.areas[contact_mask]
        contact_v0 = self.v0[contact_mask]  # 分類には元のXYを使う

        # バネっぽいスコア:
        # d = 0 で最大、threshold で 0 になるようなカーブ
        # S_local = area * (1 - (d/th)^2)
        th = self.contact_threshold
        weight = 1.0 - (contact_d / th) ** 2
        weight = np.clip(weight, 0.0, 1.0)
        local_score = contact_a * weight
        contact_score = float(local_score.sum())

        # 接触面積の合計
        total_area = float(contact_a.sum())

        # 前歯(+) / 臼歯(-) の面積
        y = contact_v0[:, 1]
        anterior_area = float(contact_a[y >= self.y_mid].sum())
        posterior_area = float(contact_a[y < self.y_mid].sum())

        # 左 / 右 の面積
        x = contact_v0[:, 0]
        left_area = float(contact_a[x <= self.x_mid].sum())
        right_area = float(contact_a[x > self.x_mid].sum())

        # バランススコア（輪ゴムが偏っていないほど良い）
        ap_balance = min(anterior_area, posterior_area)
        lr_balance = min(left_area, right_area)

        balance_score = (
            self.balance_ap_weight * ap_balance +
            self.balance_lr_weight * lr_balance
        )

        # 回転・移動は大きすぎるとペナルティ
        rot_pen = self.rot_penalty * (abs(rx_rad) + abs(ry_rad))
        trans_pen = self.trans_penalty * abs(tz)

        total_score = contact_score + balance_score - rot_pen - trans_pen

        info = {
            "total_area": total_area,
            "anterior_area": anterior_area,
            "posterior_area": posterior_area,
            "left_area": left_area,
            "right_area": right_area,
            "num_contacts": int(contact_a.shape[0]),
            "rx": rx_rad,
            "ry": ry_rad,
            "tz": tz,
        }
        return total_score, info


# =============================
# 探索アルゴリズム
# =============================

def line_search_tz(scorer: SpringOcclusionScorer,
                   rx0=0.0, ry0=0.0,
                   tz_start=0.5, tz_end=-1.5, step=-0.05):
    """
    tz 方向にまっすぐ閉口しながら、スコア最大となる tz を探す
    → これをヒルクライムの初期値にする
    """
    best_score = -1e9
    best_tz = 0.0
    best_info = None

    tz = tz_start
    print("\n[Step1] tz 方向の単純スキャンで良さそうな初期位置を探索")
    i = 0
    while tz >= tz_end - 1e-9:
        score, info = scorer.evaluate(rx0, ry0, tz)
        if i % 5 == 0:
            print(f"  tz={tz:6.3f} mm -> score={score:7.3f}, area={info['total_area']:.4f}")
        if score > best_score:
            best_score = score
            best_tz = tz
            best_info = info
        tz += step
        i += 1

    print(f"\n  → 初期候補: tz={best_tz:.3f} mm, score={best_score:.3f}, "
          f"area={best_info['total_area']:.4f}")
    return best_tz, best_score, best_info


def hill_climb_3d(scorer: SpringOcclusionScorer,
                  rx_init, ry_init, tz_init,
                  deg_step=0.5, tz_step=0.05,
                  max_iter=25,
                  max_rot_deg=5.0,
                  tz_min=-2.0, tz_max=1.0):
    """
    (rx, ry, tz) の3自由度ヒルクライム
    """
    rx = rx_init
    ry = ry_init
    tz = tz_init

    score, info = scorer.evaluate(rx, ry, tz)
    print("\n[Step2] 近傍ヒルクライム開始")
    print(f"  start: rx={np.rad2deg(rx):.3f}°, ry={np.rad2deg(ry):.3f}°, "
          f"tz={tz:.3f} mm, score={score:.3f}, area={info['total_area']:.4f}")

    rad_step = np.deg2rad(deg_step)
    max_rot_rad = np.deg2rad(max_rot_deg)

    for it in range(max_iter):
        improved = False
        best_local_score = score
        best_local_params = (rx, ry, tz)
        best_local_info = info

        for d_rx in [-rad_step, 0.0, rad_step]:
            for d_ry in [-rad_step, 0.0, rad_step]:
                for d_tz in [-tz_step, 0.0, tz_step]:
                    if d_rx == 0.0 and d_ry == 0.0 and d_tz == 0.0:
                        continue

                    rx_c = rx + d_rx
                    ry_c = ry + d_ry
                    tz_c = tz + d_tz

                    # 範囲制限
                    if abs(rx_c) > max_rot_rad or abs(ry_c) > max_rot_rad:
                        continue
                    if tz_c < tz_min or tz_c > tz_max:
                        continue

                    s_c, info_c = scorer.evaluate(rx_c, ry_c, tz_c)
                    if s_c > best_local_score:
                        best_local_score = s_c
                        best_local_params = (rx_c, ry_c, tz_c)
                        best_local_info = info_c
                        improved = True

        if not improved:
            print(f"  it={it}: 改善なし → 終了")
            break

        rx, ry, tz = best_local_params
        score = best_local_score
        info = best_local_info
        print(f"  it={it+1}: rx={np.rad2deg(rx):5.2f}°, ry={np.rad2deg(ry):5.2f}°, "
              f"tz={tz:6.3f} mm, score={score:7.3f}, area={info['total_area']:.4f}, "
              f"contacts={info['num_contacts']}")

    return rx, ry, tz, score, info


# =============================
# メイン
# =============================

def main():
    print("=" * 80)
    print("咬頭嵌合位自動最適化（輪ゴムスプリングモデル・簡易高速版）")
    print("=" * 80)

    upper_path, lower_path = select_two_stl_files()
    upper = load_mesh_safely(upper_path)
    lower = load_mesh_safely(lower_path)

    # 下顎頂点のエリア & サンプリング
    print("\n頂点面積を計算中...")
    lower_vertex_area_all = per_vertex_area(lower)

    all_vertices = lower.vertices
    n_vertices = len(all_vertices)
    SAMPLE_SIZE = 1200  # 計算時間と精度のバランス（800〜2000くらいで調整可）

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
        contact_threshold=0.03,     # 0〜0.03mm を「輪ゴムが届いている範囲」とみなす
        rot_penalty=1.5,            # 回転ペナルティ
        trans_penalty=2.0,          # tzペナルティ
        balance_ap_weight=0.4,      # 前後バランスの重み
        balance_lr_weight=0.4,      # 左右バランスの重み
    )

    # Step1: tz 方向スキャンで初期位置
    best_tz, best_score_tz, info_tz = line_search_tz(
        scorer,
        rx0=0.0,
        ry0=0.0,
        tz_start=0.5,
        tz_end=-1.5,
        step=-0.05
    )

    # Step2: 近傍ヒルクライム
    rx_best, ry_best, tz_best, score_best, info_best = hill_climb_3d(
        scorer,
        rx_init=0.0,
        ry_init=0.0,
        tz_init=best_tz,
        deg_step=0.5,
        tz_step=0.05,
        max_iter=25,
        max_rot_deg=5.0,
        tz_min=-2.0,
        tz_max=1.0,
    )

    print("\n最終結果（輪ゴムスプリングモデル）")
    print("-" * 80)
    print(f"  rx = {np.rad2deg(rx_best):6.3f} °")
    print(f"  ry = {np.rad2deg(ry_best):6.3f} °")
    print(f"  tz = {tz_best:6.3f} mm")
    print(f"  score         = {score_best:.3f}")
    print(f"  total area    = {info_best['total_area']:.4f} mm²")
    print(f"  anterior area = {info_best['anterior_area']:.4f} mm²")
    print(f"  posterior area= {info_best['posterior_area']:.4f} mm²")
    print(f"  left area     = {info_best['left_area']:.4f} mm²")
    print(f"  right area    = {info_best['right_area']:.4f} mm²")
    print(f"  contacts      = {info_best['num_contacts']} points")
    print("-" * 80)

    # 下顎全体に最終変換を適用して保存
    rot_best = R.from_euler("xyz", [rx_best, ry_best, 0.0]).as_matrix()
    transformed_all = (rot_best @ lower.vertices.T).T + np.array([0.0, 0.0, tz_best])

    lower_out = lower.copy()
    lower_out.vertices = transformed_all

    out_dir = os.path.dirname(lower_path)
    lower_name = os.path.splitext(os.path.basename(lower_path))[0]
    out_path = os.path.join(out_dir, f"{lower_name}_spring_mip.stl")
    lower_out.export(out_path)
    print(f"\n✓ 最終下顎 STL を保存しました: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

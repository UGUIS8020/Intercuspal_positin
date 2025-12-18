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
    ファイルダイアログから STL ファイルを2回に分けて選択
    1回目: 上顎, 2回目: 下顎
    """
    root = Tk()
    root.withdraw()

    upper_path = filedialog.askopenfilename(
        title="【1/2】上顎の STL ファイルを選択してください",
        filetypes=[("STL files", "*.stl"), ("All files", "*.*")]
    )
    if not upper_path:
        print("エラー: 上顎 STL が選択されませんでした。")
        root.destroy()
        sys.exit(1)

    lower_path = filedialog.askopenfilename(
        title="【2/2】下顎の STL ファイルを選択してください",
        filetypes=[("STL files", "*.stl"), ("All files", "*.*")]
    )
    if not lower_path:
        print("エラー: 下顎 STL が選択されませんでした。")
        root.destroy()
        sys.exit(1)

    root.update()
    root.destroy()

    if os.path.abspath(upper_path) == os.path.abspath(lower_path):
        print("エラー: 同じ STL が2回選択されています。上顎と下顎は別ファイルを選んでください。")
        sys.exit(1)

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

    # ----------------------------
    # 姿勢評価
    # ----------------------------

    def evaluate(self, tx, rx_rad, ry_rad, tz, max_dist_clip=0.1):
        """
        姿勢 (tx, rx, ry, tz) に対するスコアを返す
        - tx: 左右方向スライド（mm）
        - rx, ry: ラジアン（X, Y軸まわりの回転）
        - tz: 垂直方向（mm）
        ty は 0 固定（前後スライドはここでは見ない）

        戻り値:
          score, info_dict
        """
        ty = 0.0

        # 剛体変換（下顎サンプル頂点）
        rot = R.from_euler("xyz", [rx_rad, ry_rad, 0.0]).as_matrix()
        transformed = (rot @ self.v0.T).T + np.array([tx, ty, tz])

        # 上顎メッシュとの最近接距離
        closest_points, distances, tri_id = self.upper.nearest.on_surface(transformed)

        # 「輪ゴムが届いている」範囲：contact_threshold 以内
        d = np.clip(distances, 0.0, max_dist_clip)
        contact_mask = d <= self.contact_threshold

        # --------------------------------------------------
        # 1) まったく噛んでいない場合
        #    → 回転・移動ペナルティ + 大きなマイナス定数
        #       （どんな「噛んでいる姿勢」より必ず不利にする）
        # --------------------------------------------------
        if not np.any(contact_mask):
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
            }
            return score, info

        # --------------------------------------------------
        # 2) ここから下は「接触あり」のケース
        # --------------------------------------------------

        # contact_mask 部だけの距離・面積
        th = self.contact_threshold
        d_c = d[contact_mask]
        w = 1.0 - (d_c / th) ** 2               # d=0 で1, d=th で0
        w = np.clip(w, 0.0, 1.0)

        # 「バネの縮み量 × 断面積」のようなイメージ
        local_strength_c = self.areas[contact_mask] * w

        # 全頂点長の配列に戻す（コンタクト頂点以外は0）
        strength_full = np.zeros_like(self.areas)
        area_full = np.zeros_like(self.areas)
        strength_full[contact_mask] = local_strength_c
        area_full[contact_mask] = self.areas[contact_mask]

        # ----- バネごとのスコア・面積 -----
        region_scores = {}
        region_areas = {}
        scores_list = []

        for name in self.valid_regions:
            mask = self.region_masks[name]
            s = float(strength_full[mask].sum())
            a = float(area_full[mask].sum())
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
        total_area = float(area_full.sum())

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
        score = (
            0.4 * total_strength   # 全体として噛んでいるか
            + 1.8 * min_region     # 一番弱いバネもちゃんと張っているか
            - 0.3 * var_region     # 強いバネと弱いバネの差が大きいほど減点
            - 0.8 * zero_regions   # 完全にサボっているブロックがあると減点
            - rot_pen
            - trans_pen
        )

        info = {
            "total_area": total_area,
            "num_contacts": int(contact_mask.sum()),
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
            print(
                f"  tz={tz:6.3f} mm -> obj={obj:7.3f}, score={score:7.3f}, "
                f"area={info['total_area']:.4f}, "
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
    SAMPLE_SIZE = 1200  # 計算時間と精度のバランス

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
        contact_threshold=0.03,  # 0〜0.03mm を「輪ゴムが届いている範囲」とみなす
        rot_penalty=1.5,
        trans_penalty=2.0,
    )

    # Step1: tz 方向スキャンで初期位置
    best_tz, best_score_tz, info_tz = line_search_tz(
        scorer,
        tx0=0.0,
        rx0=0.0,
        ry0=0.0,
        tz_start=0.5,
        tz_end=-1.5,
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
        tz_max=1.0,
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
    step=-0.01,
    max_score_drop=0.11,
    w_right_post=0.60,
    w_pml_pen=0.40,
    w_depth=0.05,
):
    base_score, base_info = scorer.evaluate(tx, rx, ry, tz_start)

    def S(info, key):
        return float(info["region_scores"].get(key, 0.0))

    # ★ tz_start を obj の基準として評価しておく（重要）
    right_post0 = S(base_info, "M_R") + S(base_info, "PM_R")
    pml0 = S(base_info, "PM_L")
    obj0 = (
        base_score
        + w_right_post * right_post0
        - w_pml_pen * pml0
        + w_depth * 0.0
    )

    best_tz = tz_start
    best_score = base_score
    best_info = base_info
    best_obj = obj0

    tz = tz_start + step
    tz_limit = tz_start - extra_depth

    print("\n[Phase2: gyu_refine_tz] tz だけ少し『ギュッ』と探索します")
    while tz >= tz_limit - 1e-9:
        score, info = scorer.evaluate(tx, rx, ry, tz)
        ra = info["region_areas"]
        rs = info["region_scores"]

        print(
            f"  tz={tz:6.3f} mm -> score={score:7.3f}, area={info['total_area']:.4f}, "
            f"[area] M_L={ra['M_L']:.3f}, M_R={ra['M_R']:.3f}, PM_L={ra['PM_L']:.3f}, PM_R={ra['PM_R']:.3f}, ANT={ra['ANT']:.3f} | "
            f"[str] M_L={rs['M_L']:.6f}, M_R={rs['M_R']:.6f}, PM_L={rs['PM_L']:.6f}, PM_R={rs['PM_R']:.6f}, ANT={rs['ANT']:.6f}"
        )

        if score >= base_score - max_score_drop:
            right_post = S(info, "M_R") + S(info, "PM_R")
            pml = S(info, "PM_L")
            depth_bonus = (tz_start - tz)

            obj = (
                score
                + w_right_post * right_post
                - w_pml_pen * pml
                + w_depth * depth_bonus
            )

            if obj > best_obj:
                best_obj = obj
                best_tz = tz
                best_score = score
                best_info = info

        tz += step

    print(f"\n  → gyu 結果: tz={best_tz:.3f} mm, score={best_score:.3f}")
    return best_tz, best_score, best_info

   

if __name__ == "__main__":
    main()

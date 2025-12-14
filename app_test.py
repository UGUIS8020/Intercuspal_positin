#ゴムの本数を変更するテスト版

import os
import sys
import numpy as np
import trimesh
from tkinter import Tk, filedialog
from scipy.spatial.transform import Rotation as R

from dataclasses import dataclass

@dataclass
class Config:
    springs: int = 6              # ← ここだけ変えればOKにする
    contact_threshold: float = 0.03
    rot_penalty: float = 1.5
    trans_penalty: float = 2.0

    sample_size: int = 1200

    tz_start: float = 0.5
    tz_end: float = -1.5
    tz_scan_step: float = -0.05

    tx_step: float = 0.05
    deg_step: float = 0.5
    tz_step: float = 0.05
    max_iter: int = 20

    tx_min: float = -0.8
    tx_max: float = 0.8
    max_rot_deg: float = 5.0
    tz_min: float = -2.0
    tz_max: float = 1.0

    gyu_extra_depth: float = 0.10
    gyu_step: float = -0.01
    gyu_max_score_drop: float = 0.11  

    @property
    def n_bands(self):
        return springs_preset(self.springs)[0]

    @property
    def merge_front(self):
        return springs_preset(self.springs)[1]  

cfg = Config() 


# =============================
# ユーティリティ
# =============================
def springs_preset(springs: int):
    """
    springs 本数 → (n_bands, merge_front) に変換
    - 4本 : 2バンド×左右
    - 5本 : 3バンドで前歯だけ左右まとめる（従来）
    - 6本 : 3バンド×左右
    - 10本: 5バンド×左右
    """
    presets = {
        4:  (2, False),
        5:  (3, True),
        6:  (3, False),
        10: (5, False),
    }
    if springs not in presets:
        raise ValueError(f"springs は {sorted(presets.keys())} のいずれかにしてください")
    return presets[springs]

def region_area_str(ra: dict, keys: list[str], digits=3):
    parts = []
    for k in keys:
        parts.append(f"{k}={ra.get(k, 0.0):.{digits}f}")
    return ", ".join(parts)

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
        mesh = trimesh.load(filepath, force='mesh')
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
    def __init__(
        self,
        upper_mesh: trimesh.Trimesh,
        lower_sample_vertices: np.ndarray,
        lower_sample_areas: np.ndarray,
        contact_threshold: float = 0.03,
        rot_penalty: float = 1.5,
        trans_penalty: float = 2.0,
        # ★追加：本数切替用
        n_bands: int = 3,
        merge_front: bool = False,   # ★6本なら False（前歯も左右を分ける）
    ):
        self.upper = upper_mesh
        self.v0 = lower_sample_vertices
        self.areas = lower_sample_areas
        self.contact_threshold = contact_threshold
        self.rot_penalty = rot_penalty
        self.trans_penalty = trans_penalty

        # ★ここでn本のマスクを作る
        self.region_masks, self.valid_regions, self.x_mid = build_region_masks(
            self.v0, n_bands=n_bands, merge_front=merge_front
        )

        print("\n[ブロック分割（輪ゴムスプリング）]")
        for name in self.region_masks.keys():
            cnt = int(self.region_masks[name].sum())
            flag = "✓" if name in self.valid_regions else "（頂点なし）"
            print(f"  {name:6s}: {cnt:4d} 点 {flag}")
        print(f"  有効バネ本数: {len(self.valid_regions)}")

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

        # ----------------------------
        # n本（valid_regions数）に依存しない特徴量に変換
        # ----------------------------
        n_active = len(scores_arr)
        eps = 1e-9

        if n_active > 0:
            mean_s = float(scores_arr.mean())
            std_s  = float(scores_arr.std())
            cv     = std_s / (mean_s + eps)                 # ばらつき（正規化）
            min_q  = float(np.quantile(scores_arr, 0.2))    # 下位20%点（minの代替）
            zero_frac = float(np.mean(scores_arr < 1e-6))   # 0の割合
            total_strength_norm = total_strength / n_active # 1本あたり平均
        else:
            mean_s = std_s = cv = min_q = zero_frac = 0.0
            total_strength_norm = 0.0

        # ----------------------------
        # 左右・前後（キーに依存しない形に）
        #  - 左右: 末尾が _L / _R のキーを合計
        #  - 前方: y中央値が最大のリージョン（=最前方）を前歯扱い
        # ----------------------------
        left_area = sum(a for k, a in region_areas.items() if k.endswith("_L"))
        right_area = sum(a for k, a in region_areas.items() if k.endswith("_R"))

        # y中央値で前方リージョンを決める（valid_regionsのみ対象）
        anterior_area = 0.0
        if len(self.valid_regions) > 0:
            medy = {}
            for name in self.valid_regions:
                m = self.region_masks[name]
                if np.any(m):
                    medy[name] = float(np.median(self.v0[m, 1]))
            if len(medy) > 0:
                front_name = max(medy, key=medy.get)
                anterior_area = float(region_areas.get(front_name, 0.0))

        posterior_area = left_area + right_area

        # 回転・移動ペナルティ
        rot_pen = self.rot_penalty * (abs(rx_rad) + abs(ry_rad))
        trans_pen = self.trans_penalty * np.sqrt(tx * tx + tz * tz)

        # ----------------------------
        # 最終スコア（n本でも比較可能）
        # ----------------------------
        score = (
            0.8 * total_strength_norm     # 全体（1本あたり）
            + 1.2 * min_q                 # 弱い側も張っているか
            - 0.8 * cv                    # ばらつきが大きいほど減点
            - 1.0 * zero_frac             # 死んでいる割合が多いほど減点
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

            # 旧指標の代わりに、新しい指標も入れておく
            "spring_mean": mean_s,
            "spring_std": std_s,
            "spring_cv": cv,
            "spring_min_q20": min_q,
            "spring_zero_frac": zero_frac,
            "n_springs": n_active,

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
                   tz_start=0.5, tz_end=-1.5, step=-0.05):
    """
    tz 方向にまっすぐ閉口しながら、スコア最大となる tz を探す
    → これをヒルクライムの初期値にする
    """
    best_score = -1e9
    best_tz = 0.0
    best_info = None

    tz = tz_start
    print("\n[Step1] tz 方向スキャンで初期位置を探索")
    i = 0
    while tz >= tz_end - 1e-9:
        score, info = scorer.evaluate(tx0, rx0, ry0, tz)
        if i % 5 == 0:
            ra = info["region_areas"]
            keys = scorer.valid_regions
            print(
                f"  tz={tz:6.3f} mm -> score={score:7.3f}, "
                f"area={info['total_area']:.4f}, "
                f"{region_area_str(ra, keys)}"
            )
        if score > best_score:
            best_score = score
            best_tz = tz
            best_info = info
        tz += step
        i += 1

    print(
        f"\n  → 初期候補: tz={best_tz:.3f} mm, score={best_score:.3f}, "
        f"area={best_info['total_area']:.4f}"
    )
    return best_tz, best_score, best_info


def hill_climb_4d(scorer: SpringOcclusionScorer,
                  tx_init, rx_init, ry_init, tz_init,
                  tx_step=0.05, deg_step=0.5, tz_step=0.05,
                  max_iter=20,
                  tx_min=-0.8, tx_max=0.8,
                  max_rot_deg=5.0,
                  tz_min=-2.0, tz_max=1.0):
    """
    (tx, rx, ry, tz) の4自由度ヒルクライム
    左右スライド + 前後・左右回転 + 上下
    """
    tx = tx_init
    rx = rx_init
    ry = ry_init
    tz = tz_init

    score, info = scorer.evaluate(tx, rx, ry, tz)
    print("\n[Step2] 近傍ヒルクライム開始")
    print(
        f"  start: tx={tx:.3f}mm, rx={np.rad2deg(rx):.3f}°, "
        f"ry={np.rad2deg(ry):.3f}°, tz={tz:.3f} mm, "
        f"score={score:.3f}, area={info['total_area']:.4f}"
    )

    rad_step = np.deg2rad(deg_step)
    max_rot_rad = np.deg2rad(max_rot_deg)

    for it in range(max_iter):
        improved = False
        best_local_score = score
        best_local_params = (tx, rx, ry, tz)
        best_local_info = info

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

                        s_c, info_c = scorer.evaluate(tx_c, rx_c, ry_c, tz_c)
                        if s_c > best_local_score:
                            best_local_score = s_c
                            best_local_params = (tx_c, rx_c, ry_c, tz_c)
                            best_local_info = info_c
                            improved = True

        if not improved:
            print(f"  it={it}: 改善なし → 終了")
            break

        tx, rx, ry, tz = best_local_params
        score = best_local_score
        info = best_local_info
        ra = info["region_areas"]
        keys = scorer.valid_regions
        print(
            f"  it={it+1}: tx={tx:6.3f}mm, rx={np.rad2deg(rx):5.2f}°, "
            f"ry={np.rad2deg(ry):5.2f}°, tz={tz:6.3f} mm, "
            f"score={score:7.3f}, area={info['total_area']:.4f}, "
            f"{region_area_str(ra, keys)}"
        )

    return tx, rx, ry, tz, score, info


# =============================
# メイン
# =============================

def main():
    print("=" * 80)
    print(f"咬頭嵌合位自動最適化（{cfg.springs}本の輪ゴムスプリングモデル）")
    print("=" * 80)

    upper_path, lower_path = select_two_stl_files()
    upper = load_mesh_safely(upper_path)
    lower = load_mesh_safely(lower_path)

    # 下顎頂点のエリア & サンプリング
    print("\n頂点面積を計算中...")
    lower_vertex_area_all = per_vertex_area(lower)

    all_vertices = lower.vertices
    n_vertices = len(all_vertices)
    SAMPLE_SIZE = cfg.sample_size

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
        contact_threshold=cfg.contact_threshold,
        rot_penalty=cfg.rot_penalty,
        trans_penalty=cfg.trans_penalty,
        n_bands=cfg.n_bands,
        merge_front=cfg.merge_front,
    )

    # Step1: tz 方向スキャンで初期位置
    best_tz, best_score_tz, info_tz = line_search_tz(
        scorer,
        tx0=0.0, rx0=0.0, ry0=0.0,
        tz_start=cfg.tz_start,
        tz_end=cfg.tz_end,
        step=cfg.tz_scan_step,
    )

    # Step2 (Phase1): 近傍ヒルクライム（tx も含めて最適化）
    tx_best, rx_best, ry_best, tz_best, score_best, info_best = hill_climb_4d(
        scorer,
        tx_init=0.0, rx_init=0.0, ry_init=0.0, tz_init=best_tz,
        tx_step=cfg.tx_step,
        deg_step=cfg.deg_step,
        tz_step=cfg.tz_step,
        max_iter=cfg.max_iter,
        tx_min=cfg.tx_min, tx_max=cfg.tx_max,
        max_rot_deg=cfg.max_rot_deg,
        tz_min=cfg.tz_min, tz_max=cfg.tz_max,
    )

    print("\nPhase1 結果（ノーマル咬合位置）")
    print("-" * 80)
    print(f"  tx = {tx_best:6.3f} mm")
    print(f"  rx = {np.rad2deg(rx_best):6.3f} °")
    print(f"  ry = {np.rad2deg(ry_best):6.3f} °")
    print(f"  tz = {tz_best:6.3f} mm")
    print(f"  score           = {score_best:.3f}")
    print(f"  total area      = {info_best['total_area']:.4f} mm²")
    print(f"  contacts        = {info_best['num_contacts']} points")
    print(f"  n_springs       = {info_best['n_springs']}")
    print(f"  zero_frac       = {info_best['spring_zero_frac']:.3f}")
    print(f"  cv              = {info_best['spring_cv']:.3f}")
    print(f"  min_q20         = {info_best['spring_min_q20']:.3f}")

    ra = info_best["region_areas"]
    for k in scorer.valid_regions:
        print(f"  {k} area        = {ra.get(k,0.0):.4f} mm²")

    # ★ Phase2: tz だけを少し「ギュッ」と噛み込ませる
    tz_gyu, score_gyu, info_gyu = gyu_refine_tz(
        scorer,
        tx_best, rx_best, ry_best, tz_best,
        extra_depth=cfg.gyu_extra_depth,
        step=cfg.gyu_step,
        max_score_drop=cfg.gyu_max_score_drop,
    )

    print("\n最終結果（Phase2: ちょっとギュッ後）")
    print("-" * 80)
    print(f"  tx = {tx_best:6.3f} mm")              # tx, rx, ry は Phase1 のまま
    print(f"  rx = {np.rad2deg(rx_best):6.3f} °")
    print(f"  ry = {np.rad2deg(ry_best):6.3f} °")
    print(f"  tz = {tz_gyu:6.3f} mm")              # tz だけ gyu 版
    print(f"  score           = {score_gyu:.3f}")
    print(f"  total area      = {info_gyu['total_area']:.4f} mm²")
    print(f"  contacts        = {info_gyu['num_contacts']} points")
    print(f"  n_springs       = {info_gyu['n_springs']}")
    print(f"  zero_frac       = {info_gyu['spring_zero_frac']:.3f}")
    print(f"  cv              = {info_gyu['spring_cv']:.3f}")
    print(f"  min_q20         = {info_gyu['spring_min_q20']:.3f}")

    ra = info_gyu["region_areas"]
    for k in scorer.valid_regions:
        print(f"  {k} area        = {ra.get(k,0.0):.4f} mm²")

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
    out_path = os.path.join(out_dir, f"{lower_name}_spring{cfg.springs}_balanced_gyu.stl")  # ★ファイル名も分けておく
    lower_out.export(out_path)
    print(f"\n✓ 最終下顎 STL を保存しました: {out_path}")
    print("=" * 80)

def gyu_refine_tz(
    scorer: SpringOcclusionScorer,
    tx, rx, ry, tz_start,
    extra_depth=0.10,      # どこまで深く探索するか（mm）
    step=-0.01,            # 探索刻み
    max_score_drop=0.11,   # Phase1 からどこまでスコア低下を許容するか
):
    """
    Phase1 で決めた tx, rx, ry を固定したまま、
    tz だけを少しマイナス方向（咬み込み方向）に動かして
    「ちょっとだけギュッとした」位置を探す。

    - Phase1 のスコアから max_score_drop までの悪化は許容
    - その範囲で「最も深い tz」を採用
    """
    # Phase1 の基準スコア
    base_score, base_info = scorer.evaluate(tx, rx, ry, tz_start)

    best_tz = tz_start
    best_score = base_score
    best_info = base_info

    tz = tz_start + step
    tz_limit = tz_start - extra_depth  # 例: tz_start=-0.15, extra_depth=0.10 → -0.25 まで

    print("\n[Phase2: gyu_refine_tz] tz だけ少し『ギュッ』と探索します")
    keys = scorer.valid_regions  # ←有効なバネ名（B1_L など）が入る

    while tz >= tz_limit - 1e-9:
        score, info = scorer.evaluate(tx, rx, ry, tz)
        ra = info["region_areas"]

        print(
            f"  tz={tz:6.3f} mm -> score={score:7.3f}, "
            f"area={info['total_area']:.4f}, "
            f"{region_area_str(ra, keys)}"
        )

        # スコア低下が許容範囲内なら「候補」として受け入れる
        if score >= base_score - max_score_drop:
            if tz < best_tz:
                best_tz = tz
                best_score = score
                best_info = info

        tz += step

    print(f"\n  → gyu 結果: tz={best_tz:.3f} mm, score={best_score:.3f}")
    return best_tz, best_score, best_info

def build_region_masks(v0: np.ndarray, n_bands: int, merge_front: bool):
    x = v0[:, 0]
    y = v0[:, 1]
    x_mid = float(np.median(x))
    is_left = x <= x_mid
    is_right = ~is_left

    # y を分位点で切る（症例ごとの歯列長に追従して安定）
    qs = np.linspace(0.0, 1.0, n_bands + 1)
    ycuts = np.quantile(y, qs)

    masks = {}
    for i in range(n_bands):
        y0, y1 = ycuts[i], ycuts[i + 1]
        band = (y >= y0) & (y <= y1) if i == n_bands - 1 else ((y >= y0) & (y < y1))

        is_front = (i == n_bands - 1)  # 最前方バンド
        if merge_front and is_front:
            masks[f"B{i+1}"] = band                  # 前歯を左右まとめて 1 本
        else:
            masks[f"B{i+1}_L"] = band & is_left
            masks[f"B{i+1}_R"] = band & is_right

    valid = [k for k, m in masks.items() if np.any(m)]
    return masks, valid, x_mid


if __name__ == "__main__":
    main()

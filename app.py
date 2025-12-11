import os
import sys
import numpy as np
import trimesh
from tkinter import Tk, filedialog
from scipy.spatial.transform import Rotation as R


def select_two_stl_files():
    """
    ファイルダイアログから STL ファイルを2つだけ選択させる。
    戻り値: (upper_path, lower_path)
    ※ 1つ目: 上顎, 2つ目: 下顎 のつもりで選んでください
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
        print("（1つ目: 上顎, 2つ目: 下顎 の順で選択）")
        sys.exit(1)

    upper_path, lower_path = filepaths
    print("上顎 STL:", upper_path)
    print("下顎 STL:", lower_path)
    return upper_path, lower_path


def per_vertex_area(mesh: trimesh.Trimesh):
    """
    各三角形の面積を3頂点に等分配して、
    頂点ごとの代表面積にする簡易計算。
    """
    areas = np.zeros(len(mesh.vertices))
    for face, area in zip(mesh.faces, mesh.area_faces):
        for vid in face:
            areas[vid] += area / 3.0
    return areas


def evaluate_contact_score(
    tx, ty, rx, ry, tz,
    sample_vertices, sample_areas, upper,
    x_mid, y_mid
):
    """
    姿勢 (tx, ty, rx, ry, tz) に対する
    「接触面積 + バランスボーナス - ペナルティ」のスコアを返す。

    tx, ty, tz: 並進（mm相当）
    rx, ry:     回転（radian）
    x_mid, y_mid: 左右・前後の境界
    """
    # 回転行列（Z軸回りはここでは 0）
    rot = R.from_euler("xyz", [rx, ry, 0.0]).as_matrix()

    # サンプル頂点を回転＋平行移動
    transformed = (rot @ sample_vertices.T).T + np.array([tx, ty, tz])

    # 最近接距離（上顎→サンプル下顎）
    closest_points, distances, triangle_id = upper.nearest.on_surface(transformed)

    # 接触判定の閾値
    tol_contact = 0.02   # 0.02 mm 以内 → 接触
    tol_tight   = 0.005  # 0.005 mm 以内 → きつい接触（ペナルティ）

    contact_mask = distances <= tol_contact
    contact_idx = np.where(contact_mask)[0]

    contact_area = 0.0
    anterior_area = 0.0
    posterior_area = 0.0
    left_area = 0.0
    right_area = 0.0

    if contact_idx.size > 0:
        c_areas = sample_areas[contact_idx]
        c_pts = sample_vertices[contact_idx]

        x = c_pts[:, 0]
        y = c_pts[:, 1]

        contact_area = float(c_areas.sum())

        # 前後
        anterior_mask = y >= y_mid
        posterior_mask = y < y_mid
        anterior_area = float(c_areas[anterior_mask].sum())
        posterior_area = float(c_areas[posterior_mask].sum())

        # 左右
        left_mask = x <= x_mid
        right_mask = x > x_mid
        left_area = float(c_areas[left_mask].sum())
        right_area = float(c_areas[right_mask].sum())

    # きつすぎる接触（めり込み）ペナルティ
    tight_mask = distances <= tol_tight
    tight_area = float(sample_areas[tight_mask].sum())
    penetration_penalty = tight_area

    # 前後・左右バランスのボーナス
    ant_post_balance = min(anterior_area, posterior_area)
    left_right_balance = min(left_area, right_area)

    # 動きすぎペナルティ
    rot_penalty = abs(rx) + abs(ry)
    trans_penalty = np.linalg.norm([tx, ty, tz])

    # 重みはあとで調整できるパラメータ
    SCORE_BALANCE_AP = 0.4   # 前後バランス
    SCORE_BALANCE_LR = 0.2   # 左右バランス

    score = (
        contact_area
        + SCORE_BALANCE_AP * ant_post_balance
        + SCORE_BALANCE_LR * left_right_balance
        - 0.5 * penetration_penalty
        - 2.0 * rot_penalty
        - 3.0 * trans_penalty
    )

    return score, contact_area


def close_until_first_contact(
    sample_vertices, sample_areas, upper, x_mid, y_mid,
    rx0=0.0, ry0=0.0, tz0=0.0,
    step=-0.05, max_steps=40
):
    """
    バイト位置 (rx0, ry0, tz0) から tz 方向に少しずつ閉口していき、
    接触面積が 0 → >0 になる最初の位置を探す。
    tx, ty はここでは 0 のまま固定。
    """
    tx = 0.0
    ty = 0.0
    rx, ry, tz = rx0, ry0, tz0

    last_score, last_area = evaluate_contact_score(
        tx, ty, rx, ry, tz,
        sample_vertices, sample_areas, upper,
        x_mid, y_mid
    )

    for i in range(max_steps):
        tz_new = tz + step
        score, area = evaluate_contact_score(
            tx, ty, rx, ry, tz_new,
            sample_vertices, sample_areas, upper,
            x_mid, y_mid
        )
        print(f"[close step {i+1}] tz={tz_new:.3f}, area={area:.4f}")

        tz = tz_new
        last_score, last_area = score, area
        if area > 0.0:
            print("最初の接触が得られました。")
            return tx, ty, rx, ry, tz, score, area

    print("close_until_first_contact: 接触が見つからないまま終了しました。")
    return tx, ty, rx, ry, tz, last_score, last_area


def hill_climb_guided(
    tx_init, ty_init, rx_init, ry_init, tz_init,
    sample_vertices, sample_areas, upper, x_mid, y_mid,
    deg_step=0.5, t_step=0.05,
    max_iter=20
):
    """
    誘導面に従うイメージで、近傍姿勢の中から
    「スコアが一番良い方向」に少しずつ移動していくヒルクライム。
    5自由度: tx, ty, rx, ry, tz
    """
    tx = tx_init
    ty = ty_init
    rx = rx_init
    ry = ry_init
    tz = tz_init

    score, area = evaluate_contact_score(
        tx, ty, rx, ry, tz,
        sample_vertices, sample_areas, upper,
        x_mid, y_mid
    )
    print(
        f"[hill start] tx={tx:.3f}, ty={ty:.3f}, "
        f"rx={np.rad2deg(rx):.3f}°, ry={np.rad2deg(ry):.3f}°, "
        f"tz={tz:.3f}, area={area:.4f}"
    )

    rad_step = np.deg2rad(deg_step)

    for it in range(max_iter):
        best_local_score = score
        best_local_params = (tx, ty, rx, ry, tz)

        # 近傍：各自由度を [-step, 0, +step] でゆらす
        for d_tx in [-t_step, 0.0, t_step]:
            for d_ty in [-t_step, 0.0, t_step]:
                for d_rx in [-rad_step, 0.0, rad_step]:
                    for d_ry in [-rad_step, 0.0, rad_step]:
                        for d_tz in [-t_step, 0.0, t_step]:

                            if (
                                d_tx == 0.0 and d_ty == 0.0 and
                                d_rx == 0.0 and d_ry == 0.0 and d_tz == 0.0
                            ):
                                continue

                            tx_c = tx + d_tx
                            ty_c = ty + d_ty
                            rx_c = rx + d_rx
                            ry_c = ry + d_ry
                            tz_c = tz + d_tz

                            # 動きの範囲制限
                            if abs(rx_c) > np.deg2rad(5.0):
                                continue
                            if abs(ry_c) > np.deg2rad(5.0):
                                continue
                            if abs(tx_c) > 0.6 or abs(ty_c) > 0.6:
                                continue
                            if tz_c < -1.5 or tz_c > 1.0:
                                continue

                            s, a = evaluate_contact_score(
                                tx_c, ty_c, rx_c, ry_c, tz_c,
                                sample_vertices, sample_areas, upper,
                                x_mid, y_mid
                            )

                            if s > best_local_score:
                                best_local_score = s
                                best_local_params = (tx_c, ty_c, rx_c, ry_c, tz_c)

        # 改善がなければ終了（局所最大）
        if best_local_score <= score:
            print(f"[hill stop] it={it}, 改善なし → 終了")
            break

        tx, ty, rx, ry, tz = best_local_params
        score = best_local_score
        _, area = evaluate_contact_score(
            tx, ty, rx, ry, tz,
            sample_vertices, sample_areas, upper,
            x_mid, y_mid
        )

        print(
            f"[hill {it+1}] tx={tx:.3f}, ty={ty:.3f}, "
            f"rx={np.rad2deg(rx):.3f}°, ry={np.rad2deg(ry):.3f}°, "
            f"tz={tz:.3f}, score={score:.4f}, area={area:.4f}"
        )

    return tx, ty, rx, ry, tz, score, area


def main():
    # 1. STL を 2つ選択（上顎・下顎）
    upper_path, lower_path = select_two_stl_files()

    # 2. メッシュ読み込み
    upper = trimesh.load(upper_path)  # 上顎：固定
    lower = trimesh.load(lower_path)  # 下顎：バイト位置

    # 3. 頂点ごとの代表面積（下顎・全頂点）
    lower_vertex_area_all = per_vertex_area(lower)

    # 3.1 計算用に頂点をサンプリング
    all_vertices = lower.vertices
    n_vertices = len(all_vertices)
    print(f"下顎の頂点数: {n_vertices}")

    SAMPLE_SIZE = 2000
    if n_vertices > SAMPLE_SIZE:
        rng = np.random.default_rng(0)
        sample_idx = rng.choice(n_vertices, size=SAMPLE_SIZE, replace=False)
    else:
        sample_idx = np.arange(n_vertices)

    sample_vertices = all_vertices[sample_idx]
    sample_areas = lower_vertex_area_all[sample_idx]

    print(f"サンプリングする頂点数: {len(sample_vertices)}")

    # 前後・左右の境界（中央値でざっくり分ける）
    sample_x = sample_vertices[:, 0]
    sample_y = sample_vertices[:, 1]
    x_mid = float(np.median(sample_x))
    y_mid = float(np.median(sample_y))
    print(f"x_mid (左右の境界) = {x_mid:.4f}")
    print(f"y_mid (前後の境界) = {y_mid:.4f}")

    # 4. ステージ1：軽く沈めて最初の接触を作る（tx, ty は 0 のまま）
    tx0, ty0, rx0, ry0, tz0, score0, area0 = close_until_first_contact(
        sample_vertices, sample_areas, upper, x_mid, y_mid,
        rx0=0.0, ry0=0.0, tz0=0.0,
        step=-0.05, max_steps=40
    )

    # 5. ステージ2：誘導に沿ってヒルクライム
    tx_best, ty_best, rx_best, ry_best, tz_best, score_best, area_best = hill_climb_guided(
        tx0, ty0, rx0, ry0, tz0,
        sample_vertices, sample_areas, upper, x_mid, y_mid,
        deg_step=0.5,   # 回転ステップ
        t_step=0.05,    # 並進ステップ
        max_iter=20
    )

    print("最終結果:")
    print("  tx, ty [mm]  =", tx_best, ty_best)
    print("  rx, ry [deg] =", np.rad2deg(rx_best), np.rad2deg(ry_best))
    print("  tz [mm]      =", tz_best)
    print("  score        =", score_best)
    print("  contact area =", area_best)

    # 6. ベストな回転＋平行移動を下顎全体に適用
    rot_best = R.from_euler("xyz", [rx_best, ry_best, 0.0]).as_matrix()
    best_vertices = (rot_best @ lower.vertices.T).T + np.array([tx_best, ty_best, tz_best])

    lower_refined = lower.copy()
    lower_refined.vertices = best_vertices

    # 保存先は下顎ファイルと同じフォルダ
    lower_dir = os.path.dirname(lower_path)
    lower_name = os.path.splitext(os.path.basename(lower_path))[0]
    out_path = os.path.join(lower_dir, f"{lower_name}_refined_mip_guided.stl")

    lower_refined.export(out_path)
    print("誘導に沿って調整した下顎 STL を保存しました:", out_path)


if __name__ == "__main__":
    main()

import os
import sys
import numpy as np
import trimesh
from tkinter import Tk, filedialog
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from datetime import datetime


class OptimizationConfig:
    """最適化パラメータの設定クラス"""
    # 接触判定
    CONTACT_THRESHOLD = 0.02  # mm
    TIGHT_THRESHOLD = 0.005   # mm
    
    # サンプリング
    SAMPLE_SIZE = 2000
    
    # 最適化パラメータ
    MAX_ROTATION = 5.0  # degrees
    MAX_TRANSLATION = 0.6  # mm
    ROTATION_STEP = 0.5  # degrees
    TRANSLATION_STEP = 0.05  # mm
    MAX_ITERATIONS = 20
    
    # スコア重み
    BALANCE_AP_WEIGHT = 0.4
    BALANCE_LR_WEIGHT = 0.2
    PENETRATION_PENALTY = 0.5
    ROTATION_PENALTY = 2.0
    TRANSLATION_PENALTY = 3.0
    
    # 初期閉口
    CLOSE_STEP = -0.05  # mm
    MAX_CLOSE_STEPS = 40


def select_two_stl_files():
    """
    ファイルダイアログから STL ファイルを2つだけ選択させる。
    戻り値: (upper_path, lower_path)
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


def load_mesh_safely(filepath):
    """安全なメッシュ読み込みとバリデーション"""
    try:
        mesh = trimesh.load(filepath)
        
        # メッシュの有効性チェック
        if not mesh.is_watertight:
            print(f"警告: {os.path.basename(filepath)} は水密ではありません")
        
        if len(mesh.vertices) < 100:
            raise ValueError(f"頂点数が少なすぎます: {len(mesh.vertices)}")
        
        print(f"✓ {os.path.basename(filepath)} 読み込み成功 ({len(mesh.vertices)} 頂点)")
        return mesh
        
    except Exception as e:
        print(f"エラー: {filepath} の読み込みに失敗しました")
        print(f"詳細: {e}")
        sys.exit(1)


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
    x_mid, y_mid, config
):
    """
    姿勢 (tx, ty, rx, ry, tz) に対する
    「接触面積 + バランスボーナス - ペナルティ」のスコアを返す。
    """
    # 回転行列（Z軸回りはここでは 0）
    rot = R.from_euler("xyz", [rx, ry, 0.0]).as_matrix()

    # サンプル頂点を回転＋平行移動
    transformed = (rot @ sample_vertices.T).T + np.array([tx, ty, tz])

    # 最近接距離（上顎→サンプル下顎）
    closest_points, distances, triangle_id = upper.nearest.on_surface(transformed)

    # 接触判定
    contact_mask = distances <= config.CONTACT_THRESHOLD
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
    tight_mask = distances <= config.TIGHT_THRESHOLD
    tight_area = float(sample_areas[tight_mask].sum())
    penetration_penalty = tight_area

    # 前後・左右バランスのボーナス
    ant_post_balance = min(anterior_area, posterior_area)
    left_right_balance = min(left_area, right_area)

    # 動きすぎペナルティ
    rot_penalty = abs(rx) + abs(ry)
    trans_penalty = np.linalg.norm([tx, ty, tz])

    score = (
        contact_area
        + config.BALANCE_AP_WEIGHT * ant_post_balance
        + config.BALANCE_LR_WEIGHT * left_right_balance
        - config.PENETRATION_PENALTY * penetration_penalty
        - config.ROTATION_PENALTY * rot_penalty
        - config.TRANSLATION_PENALTY * trans_penalty
    )

    # 詳細情報を返す
    details = {
        'contact_area': contact_area,
        'anterior_area': anterior_area,
        'posterior_area': posterior_area,
        'left_area': left_area,
        'right_area': right_area,
        'distances': distances
    }

    return score, details


def close_until_first_contact(
    sample_vertices, sample_areas, upper, x_mid, y_mid, config,
    rx0=0.0, ry0=0.0, tz0=0.0
):
    """
    バイト位置から tz 方向に少しずつ閉口していき、
    接触面積が 0 → >0 になる最初の位置を探す。
    """
    tx = 0.0
    ty = 0.0
    rx, ry, tz = rx0, ry0, tz0

    print("\n[ステージ1: 初期接触の確立]")
    
    for i in tqdm(range(config.MAX_CLOSE_STEPS), desc="閉口中"):
        tz_new = tz + config.CLOSE_STEP
        score, details = evaluate_contact_score(
            tx, ty, rx, ry, tz_new,
            sample_vertices, sample_areas, upper,
            x_mid, y_mid, config
        )
        
        tz = tz_new
        if details['contact_area'] > 0.0:
            print(f"\n✓ 初期接触確立: tz={tz:.3f}mm, 接触面積={details['contact_area']:.4f}mm²")
            return tx, ty, rx, ry, tz, score, details

    print("\n警告: 接触が見つかりませんでした")
    return tx, ty, rx, ry, tz, score, details


def optimize_with_scipy(
    tx_init, ty_init, rx_init, ry_init, tz_init,
    sample_vertices, sample_areas, upper, x_mid, y_mid, config
):
    """scipy.optimizeを使った効率的な最適化"""
    
    print("\n[ステージ2: 最適化（scipy.optimize）]")
    
    def objective(params):
        tx, ty, rx, ry, tz = params
        score, _ = evaluate_contact_score(
            tx, ty, rx, ry, tz,
            sample_vertices, sample_areas, upper,
            x_mid, y_mid, config
        )
        return -score  # 最小化問題に変換
    
    # 初期値
    x0 = np.array([tx_init, ty_init, rx_init, ry_init, tz_init])
    
    # 制約
    bounds = [
        (-config.MAX_TRANSLATION, config.MAX_TRANSLATION),  # tx
        (-config.MAX_TRANSLATION, config.MAX_TRANSLATION),  # ty
        (-np.deg2rad(config.MAX_ROTATION), np.deg2rad(config.MAX_ROTATION)),  # rx
        (-np.deg2rad(config.MAX_ROTATION), np.deg2rad(config.MAX_ROTATION)),  # ry
        (-1.5, 1.0)   # tz
    ]
    
    # 最適化実行
    print("最適化を実行中...")
    result = minimize(
        objective, 
        x0, 
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 100, 'disp': True}
    )
    
    if result.success:
        print("✓ 最適化成功")
    else:
        print("⚠ 最適化は収束しませんでしたが、現在の最良解を使用します")
    
    tx, ty, rx, ry, tz = result.x
    score, details = evaluate_contact_score(
        tx, ty, rx, ry, tz,
        sample_vertices, sample_areas, upper,
        x_mid, y_mid, config
    )
    
    return tx, ty, rx, ry, tz, score, details


def visualize_results(lower_refined, upper, results, output_dir):
    """結果の可視化"""
    print("\n結果を可視化中...")
    
    fig = plt.figure(figsize=(15, 5))
    
    # 3D表示 - 下顎
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_trisurf(
        lower_refined.vertices[:, 0],
        lower_refined.vertices[:, 1],
        lower_refined.vertices[:, 2],
        triangles=lower_refined.faces,
        color='lightblue',
        alpha=0.7,
        edgecolor='none'
    )
    ax1.set_title('Adjusted Lower Jaw', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    
    # 3D表示 - 上顎
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_trisurf(
        upper.vertices[:, 0],
        upper.vertices[:, 1],
        upper.vertices[:, 2],
        triangles=upper.faces,
        color='lightcoral',
        alpha=0.7,
        edgecolor='none'
    )
    ax2.set_title('Upper Jaw (Fixed)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_zlabel('Z (mm)')
    
    # 接触距離の分布
    ax3 = fig.add_subplot(133)
    distances = results['distances']
    ax3.hist(distances, bins=50, edgecolor='black', alpha=0.7)
    ax3.axvline(x=0.02, color='r', linestyle='--', linewidth=2, label='Contact threshold (0.02mm)')
    ax3.axvline(x=0.005, color='orange', linestyle='--', linewidth=2, label='Tight threshold (0.005mm)')
    ax3.set_xlabel('Distance (mm)', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.set_title('Contact Distance Distribution', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    viz_path = os.path.join(output_dir, 'contact_analysis.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"✓ 可視化結果を保存: {viz_path}")
    plt.close()


def generate_report(params, results, config, output_dir):
    """詳細レポートの生成"""
    tx, ty, rx, ry, tz = params
    
    # バランス率の計算
    total_contact = results['contact_area']
    if total_contact > 0:
        ap_balance = (min(results['anterior_area'], results['posterior_area']) / 
                     (max(results['anterior_area'], results['posterior_area']) + 1e-6)) * 100
        lr_balance = (min(results['left_area'], results['right_area']) / 
                     (max(results['left_area'], results['right_area']) + 1e-6)) * 100
    else:
        ap_balance = 0.0
        lr_balance = 0.0
    
    report = f"""
{'='*70}
咬合位置最適化レポート
{'='*70}
生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

【最終パラメータ】
  水平移動 (X軸): {tx:7.3f} mm
  水平移動 (Y軸): {ty:7.3f} mm
  垂直移動 (Z軸): {tz:7.3f} mm
  回転 (X軸):     {np.rad2deg(rx):7.2f} °
  回転 (Y軸):     {np.rad2deg(ry):7.2f} °

【接触評価】
  総接触面積:     {total_contact:7.2f} mm²
  前歯部接触:     {results['anterior_area']:7.2f} mm²
  臼歯部接触:     {results['posterior_area']:7.2f} mm²
  左側接触:       {results['left_area']:7.2f} mm²
  右側接触:       {results['right_area']:7.2f} mm²

【バランス評価】
  前後バランス率: {ap_balance:7.1f} %
  左右バランス率: {lr_balance:7.1f} %

【最適化設定】
  接触閾値:       {config.CONTACT_THRESHOLD} mm
  サンプル数:     {config.SAMPLE_SIZE} 頂点
  最大回転制限:   ±{config.MAX_ROTATION} °
  最大移動制限:   ±{config.MAX_TRANSLATION} mm

{'='*70}
"""
    
    report_path = os.path.join(output_dir, "optimization_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✓ レポート保存: {report_path}")
    
    # コンソールにも表示
    print(report)


def main():
    print("="*70)
    print("咬合位置自動最適化プログラム v2.0")
    print("="*70)
    
    config = OptimizationConfig()
    
    # 1. STL を 2つ選択（上顎・下顎）
    upper_path, lower_path = select_two_stl_files()
    
    # 出力ディレクトリ
    output_dir = os.path.dirname(lower_path)

    # 2. メッシュ読み込み
    print("\nメッシュを読み込み中...")
    upper = load_mesh_safely(upper_path)  # 上顎：固定
    lower = load_mesh_safely(lower_path)  # 下顎：バイト位置

    # 3. 頂点ごとの代表面積（下顎・全頂点）
    print("\n頂点面積を計算中...")
    lower_vertex_area_all = per_vertex_area(lower)

    # 3.1 計算用に頂点をサンプリング
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

    # 前後・左右の境界（中央値でざっくり分ける）
    sample_x = sample_vertices[:, 0]
    sample_y = sample_vertices[:, 1]
    x_mid = float(np.median(sample_x))
    y_mid = float(np.median(sample_y))
    print(f"  左右の境界 (x_mid) = {x_mid:.4f} mm")
    print(f"  前後の境界 (y_mid) = {y_mid:.4f} mm")

    # 4. ステージ1：軽く沈めて最初の接触を作る
    tx0, ty0, rx0, ry0, tz0, score0, details0 = close_until_first_contact(
        sample_vertices, sample_areas, upper, x_mid, y_mid, config,
        rx0=0.0, ry0=0.0, tz0=0.0
    )

    # 5. ステージ2：scipy.optimizeで最適化
    tx_best, ty_best, rx_best, ry_best, tz_best, score_best, details_best = optimize_with_scipy(
        tx0, ty0, rx0, ry0, tz0,
        sample_vertices, sample_areas, upper, x_mid, y_mid, config
    )

    # 6. ベストな回転＋平行移動を下顎全体に適用
    print("\n下顎メッシュを変換中...")
    rot_best = R.from_euler("xyz", [rx_best, ry_best, 0.0]).as_matrix()
    best_vertices = (rot_best @ lower.vertices.T).T + np.array([tx_best, ty_best, tz_best])

    lower_refined = lower.copy()
    lower_refined.vertices = best_vertices

    # 保存先
    lower_name = os.path.splitext(os.path.basename(lower_path))[0]
    out_path = os.path.join(output_dir, f"{lower_name}_refined_optimized.stl")

    lower_refined.export(out_path)
    print(f"✓ 最適化された下顎STLを保存: {out_path}")

    # 7. 可視化
    visualize_results(lower_refined, upper, details_best, output_dir)

    # 8. 詳細レポート生成
    params = (tx_best, ty_best, rx_best, ry_best, tz_best)
    generate_report(params, details_best, config, output_dir)

    print("\n" + "="*70)
    print("処理が完了しました！")
    print("="*70)


if __name__ == "__main__":
    main()
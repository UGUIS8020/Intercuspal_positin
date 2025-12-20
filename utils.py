import os, sys
import trimesh
import numpy as np
import open3d as o3d

# ============================================================
# ★ Decimation 設定（ここだけ触ればOK）
# ============================================================
DECIMATE_ENABLED = True
DECIMATE_REDUCTION = 0    # 0.30 → 30%削減（面数を70%に）
DECIMATE_MIN_FACES = 1000         # 制限いらないなら 0 でOK
DECIMATE_VERBOSE = True


def _as_trimesh(mesh_or_scene):
    """trimesh.load が Scene を返す場合があるので Trimesh に統一する"""
    if isinstance(mesh_or_scene, trimesh.Trimesh):
        return mesh_or_scene
    if isinstance(mesh_or_scene, trimesh.Scene):
        geoms = list(mesh_or_scene.geometry.values())
        if len(geoms) == 0:
            raise ValueError("Scene 内に geometry がありません")
        if len(geoms) == 1:
            return geoms[0]
        return trimesh.util.concatenate(geoms)
    raise TypeError(f"Unsupported mesh type: {type(mesh_or_scene)}")


def _trimesh_to_open3d(trimesh_mesh):
    """Trimesh → Open3D 変換"""
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(trimesh_mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces)
    return o3d_mesh


def _open3d_to_trimesh(o3d_mesh):
    """Open3D → Trimesh 変換"""
    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def decimate_mesh_by_reduction(mesh: trimesh.Trimesh,
                               reduction: float,
                               min_faces: int = 0,
                               verbose: bool = True) -> trimesh.Trimesh:
    """
    Open3Dを使用してメッシュを簡略化
    reduction=0.30 → 面数を70%にする（30%削減）
    """
    if not (0.0 < reduction < 1.0):
        return mesh

    original_faces = int(len(mesh.faces))
    if original_faces <= max(0, int(min_faces)):
        if verbose:
            print(f"✓ decimate skip: faces={original_faces} <= min_faces={min_faces}")
        return mesh

    target_faces = int(original_faces * (1.0 - reduction))
    target_faces = max(target_faces, 10)  # 最低限

    if verbose:
        print(f"🔧 decimate: faces {original_faces:,} → {target_faces:,} (reduction={reduction:.0%})")

    try:
        # Trimesh → Open3D
        o3d_mesh = _trimesh_to_open3d(mesh)
        
        # Open3Dで簡略化（Quadric Error Metrics）
        simplified_o3d = o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
        
        # Open3D → Trimesh
        simplified = _open3d_to_trimesh(simplified_o3d)
        
        if simplified is None or len(simplified.faces) == 0 or len(simplified.vertices) == 0:
            raise RuntimeError("simplify returned empty mesh")

        if verbose:
            print(f"✓ decimate done: faces={len(simplified.faces):,}, verts={len(simplified.vertices):,}")
        return simplified

    except Exception as e:
        if verbose:
            print("⚠ decimate failed, returning original mesh")
            print("   reason:", repr(e))
        return mesh


def load_mesh_safely(filepath):
    """trimesh で STL を読み込む（簡易チェック＋任意の簡略化付き）"""
    try:
        mesh = trimesh.load(filepath, process=True)
        mesh = _as_trimesh(mesh)

        if len(mesh.vertices) < 100:
            raise ValueError(f"頂点数が少なすぎます: {len(mesh.vertices)}")

        # ★ここで簡略化（任意）
        if DECIMATE_ENABLED and DECIMATE_REDUCTION > 0:
            mesh = decimate_mesh_by_reduction(
                mesh,
                reduction=DECIMATE_REDUCTION,
                min_faces=DECIMATE_MIN_FACES,
                verbose=DECIMATE_VERBOSE
            )

        # 水密チェック（簡略化後）
        is_watertight = mesh.is_watertight
        if not is_watertight:
            print(f"\n{'='*70}")
            print(f"⚠️  重要警告: {os.path.basename(filepath)} は水密ではありません")
            print(f"{'='*70}")
            print(f"\n【注意】本プログラムは継続しますが、結果の信頼性に注意してください")
            print(f"{'='*70}\n")

        status = "✓" if is_watertight else "⚠"
        watertight_str = "水密" if is_watertight else "非水密"
        print(f"{status} {os.path.basename(filepath)} 読み込み "
              f"({len(mesh.vertices):,} 頂点, {len(mesh.faces):,} 面, {watertight_str})")

        return mesh

    except Exception as e:
        print(f"エラー: {filepath} の読み込みに失敗しました")
        print("詳細:", e)
        sys.exit(1)
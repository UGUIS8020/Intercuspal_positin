# Copilot Instructions for Intercuspal_positin

## Project Overview

This project is a dental mesh analysis toolkit focused on occlusal (bite) analysis, supporting both full-arch and partial-arch workflows. It processes STL mesh files, computes contact points, and provides robust GPU-accelerated and CPU fallback distance calculations. The codebase is primarily in Python and uses libraries such as `trimesh`, `numpy`, `scipy`, `cupy`, and `open3d`.

## Key Components

-   **app_gyu.py**: Main logic for mesh processing, region segmentation, pivot calculation, and robust distance metrics. Contains user dialogs for workflow selection (full/partial arch, jaw selection, STL file selection).
-   **utils.py**: Mesh loading, decimation, and watertightness checks. Handles Open3D/trimesh conversions and mesh simplification.
-   **lower*sample_1200*\*.npz**: Example data files (likely mesh or point cloud samples).

## Developer Workflows

-   **Run main analysis**: Use `app_gyu.py` as the entry point. It will prompt for STL files and workflow options via GUI dialogs.
-   **Mesh loading**: Uses `utils.load_mesh_safely()` for robust STL import and optional decimation. Non-watertight meshes are allowed but warned.
-   **GPU acceleration**: If `cupy` is installed, GPU is used for distance calculations; otherwise, CPU fallback is automatic.
-   **Mesh decimation**: Controlled by constants in `utils.py` (`DECIMATE_ENABLED`, `DECIMATE_REDUCTION`). Adjust here for performance/quality tradeoff.

## Conventions & Patterns

-   **Dialog-driven workflow**: User choices (arch mode, jaw, file selection) are handled via Tkinter dialogs for reproducibility.
-   **Partial arch support**: Functions like `divide_partial_arch_regions` and `calculate_partial_arch_pivot` handle left/right segmentation and region masks.
-   **Error handling**: Most errors print detailed messages and exit; mesh loading is robust to non-watertight files but warns users.
-   **No test/ directory code**: As of this writing, test directories are present but contain no Python files.

## Integration & Dependencies

-   **requirements.txt**: Lists all dependencies. `cupy-cuda13x` is optional for GPU, others are required.
-   **No build system**: Run scripts directly with Python 3.10+.
-   **No explicit test suite**: Add tests in the `test/` directories if needed.

## Examples

-   To process meshes, run:
    ```sh
    python app_gyu.py
    ```
    and follow the GUI prompts.
-   To adjust mesh decimation, edit `DECIMATE_REDUCTION` in `utils.py`.

## Key Files

-   [app_gyu.py](../app_gyu.py): Main workflow and algorithms
-   [utils.py](../utils.py): Mesh utilities and decimation
-   [requirements.txt](../requirements.txt): Dependencies

---

If any conventions or workflows are unclear, please request clarification or examples from the maintainers.

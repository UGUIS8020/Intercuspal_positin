# gpu_stress.py
import time
import math

import torch


def run_matmul_test(size: int, iters: int, device: torch.device) -> None:
    print(f"\n=== MatMul ベンチマーク: size={size} x {size}, iters={iters} ===")
    print(f"デバイス: {device} ({torch.cuda.get_device_name(device.index)})")

    # メモリ不足に備えて try/except
    try:
        a = torch.randn((size, size), device=device)
        b = torch.randn((size, size), device=device)
    except RuntimeError as e:
        print(f"行列の確保に失敗しました (おそらくメモリ不足): {e}")
        return

    # ウォームアップ
    print("ウォームアップ中...")
    c = a @ b
    torch.cuda.synchronize()

    # ベンチ本番
    print("計測中...")
    start = time.perf_counter()
    for _ in range(iters):
        c = a @ b
    torch.cuda.synchronize()
    end = time.perf_counter()

    elapsed = end - start
    avg_time = elapsed / iters

    # FLOPs 計算 (行列積 N×N は約 2*N^3 FLOPs)
    flops_per_iter = 2 * (size ** 3)
    total_flops = flops_per_iter * iters
    tflops = total_flops / elapsed / 1e12

    print(f"総時間: {elapsed:.4f} 秒")
    print(f"1 回あたり平均時間: {avg_time:.6f} 秒")
    print(f"理論演算回数: {total_flops:.3e} FLOPs")
    print(f"実効性能: {tflops:.2f} TFLOPS (目安)")


def main():
    if not torch.cuda.is_available():
        print("CUDA GPU が利用できません。")
        return

    device = torch.device("cuda:0")
    print("=== GPU ストレステスト (行列積ベンチ) ===")
    print("PyTorch version:", torch.__version__)
    print("CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(device.index))

    # テストパターン
    # RTX 3060 ならこのくらいは概ね大丈夫なはず
    test_cases = [
        (8192, 50),   # 今の 8192x8192 の5倍の回数
        (10000, 20),  # VRAM と発熱に注意
    ]

    for size, iters in test_cases:
        run_matmul_test(size, iters, device)


if __name__ == "__main__":
    main()

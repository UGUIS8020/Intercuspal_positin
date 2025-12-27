#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI版 咬合位置最適化プログラム
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
import sys
import io
import os
from pathlib import Path
import subprocess

# 既存のロジックをインポート
try:
    from app_gyu import (
        load_mesh_safely,
        SpringOcclusionScorer,
        per_vertex_area
    )
    import numpy as np
    import trimesh
    IMPORTS_OK = True
except ImportError as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)


class TextRedirector(io.StringIO):
    """標準出力をテキストウィジェットにリダイレクト"""
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        
    def write(self, string):
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)
        self.text_widget.update_idletasks()
        
    def flush(self):
        pass


class OcclusionOptimizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("咬合位置最適化システム")
        self.root.geometry("900x700")
        
        # インポートチェック
        if not IMPORTS_OK:
            messagebox.showerror("インポートエラー", 
                f"必要なモジュールの読み込みに失敗しました:\n{IMPORT_ERROR}")
            root.destroy()
            return
        
        # 変数の初期化
        self.upper_stl_path = tk.StringVar()
        self.lower_stl_path = tk.StringVar()
        self.output_dir = tk.StringVar(value=str(Path.cwd()))
        
        # パラメータ変数
        self.move_mode = tk.StringVar(value="lower")
        self.sample_size = tk.IntVar(value=1200)
        # 全顎/片顎モード
        self.arch_mode = tk.StringVar(value="full")  # "full" or "partial"
        self.arch_side = tk.StringVar(value="right")  # "right" or "left"
        
        self.is_running = False
        self.process = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # メインフレーム
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        row = 0
        # 動かす顎の選択
        move_frame = ttk.Frame(main_frame)
        move_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 5))
        ttk.Label(move_frame, text="動かす顎:").grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(move_frame, text="下顎（Lower）", variable=self.move_mode, value="lower").grid(row=0, column=1, padx=5)
        ttk.Radiobutton(move_frame, text="上顎（Upper）", variable=self.move_mode, value="upper").grid(row=0, column=2, padx=5)
        row += 1

        # ファイル選択セクション
        ttk.Label(main_frame, text="STLファイル選択", font=("", 12, "bold")).grid(
            row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))
        row += 1

        # 上顎STL
        ttk.Label(main_frame, text="上顎STL:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(main_frame, textvariable=self.upper_stl_path, width=50).grid(
            row=row, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(main_frame, text="参照...", command=self.browse_upper_stl).grid(
            row=row, column=2)
        row += 1
        
        # 下顎STL
        ttk.Label(main_frame, text="下顎STL:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(main_frame, textvariable=self.lower_stl_path, width=50).grid(
            row=row, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(main_frame, text="参照...", command=self.browse_lower_stl).grid(
            row=row, column=2)
        row += 1
        
        # 出力ディレクトリ
        ttk.Label(main_frame, text="出力先:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(main_frame, textvariable=self.output_dir, width=50).grid(
            row=row, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(main_frame, text="参照...", command=self.browse_output_dir).grid(
            row=row, column=2)
        row += 1
        
        # パラメータセクション
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        row += 1

        ttk.Label(main_frame, text="最適化パラメータ", font=("", 12, "bold")).grid(
            row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))
        row += 1

        # 全顎/片顎モード選択
        arch_frame = ttk.Frame(main_frame)
        arch_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 5))
        ttk.Label(arch_frame, text="モード:").grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(arch_frame, text="全顎", variable=self.arch_mode, value="full", command=self.update_arch_mode).grid(row=0, column=1, padx=5)
        ttk.Radiobutton(arch_frame, text="片顎", variable=self.arch_mode, value="partial", command=self.update_arch_mode).grid(row=0, column=2, padx=5)
        # 片顎時の左右選択
        self.arch_side_frame = ttk.Frame(arch_frame)
        self.arch_side_frame.grid(row=0, column=3, padx=10)
        ttk.Label(self.arch_side_frame, text="側:").grid(row=0, column=0, sticky=tk.W)
        self.right_rb = ttk.Radiobutton(self.arch_side_frame, text="右側", variable=self.arch_side, value="right")
        self.right_rb.grid(row=0, column=1, padx=2)
        self.left_rb = ttk.Radiobutton(self.arch_side_frame, text="左側", variable=self.arch_side, value="left")
        self.left_rb.grid(row=0, column=2, padx=2)

        # 初期状態は非表示
        self.arch_side_frame.grid_remove()
        row += 1

        # ログ表示
        ttk.Label(main_frame, text="実行ログ", font=("", 12, "bold")).grid(
            row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 5))
        row += 1

        self.log_text = scrolledtext.ScrolledText(main_frame, height=20, width=80)
        self.log_text.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))

        main_frame.rowconfigure(row, weight=1)
        row += 1

        # 実行・停止ボタン
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        row += 1
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=row, column=0, columnspan=3)
        self.run_button = ttk.Button(button_frame, text="最適化実行", command=self.run_optimization, width=20)
        self.run_button.grid(row=0, column=0, padx=5)
        self.stop_button = ttk.Button(button_frame, text="停止", command=self.stop_optimization, width=20, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=5)
        row += 1

        # プログレスバー
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        row += 1

        # ログにリダイレクト
        sys.stdout = TextRedirector(self.log_text)
        sys.stderr = TextRedirector(self.log_text)

        # archモード初期表示を反映
        self.update_arch_mode()

    def update_arch_mode(self):
        if self.arch_mode.get() == "partial":
            self.arch_side_frame.grid()
        else:
            self.arch_side_frame.grid_remove()
        
        # ログにリダイレクト
        sys.stdout = TextRedirector(self.log_text)
        sys.stderr = TextRedirector(self.log_text)
        
    def browse_upper_stl(self):
        filename = filedialog.askopenfilename(
            title="上顎STLファイルを選択",
            filetypes=[("STL files", "*.stl"), ("All files", "*.*")]
        )
        if filename:
            self.upper_stl_path.set(filename)
            
    def browse_lower_stl(self):
        filename = filedialog.askopenfilename(
            title="下顎STLファイルを選択",
            filetypes=[("STL files", "*.stl"), ("All files", "*.*")]
        )
        if filename:
            self.lower_stl_path.set(filename)
            
    def browse_output_dir(self):
        dirname = filedialog.askdirectory(title="出力ディレクトリを選択")
        if dirname:
            self.output_dir.set(dirname)
            
    def validate_inputs(self):
        if not self.upper_stl_path.get():
            messagebox.showerror("エラー", "上顎STLファイルを選択してください")
            return False
        if not self.lower_stl_path.get():
            messagebox.showerror("エラー", "下顎STLファイルを選択してください")
            return False
        if not os.path.exists(self.upper_stl_path.get()):
            messagebox.showerror("エラー", "上顎STLファイルが見つかりません")
            return False
        if not os.path.exists(self.lower_stl_path.get()):
            messagebox.showerror("エラー", "下顎STLファイルが見つかりません")
            return False
        return True
        
    def run_optimization(self):
        if not self.validate_inputs():
            return
            
        self.is_running = True
        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.progress.start(10)
        self.log_text.delete(1.0, tk.END)
        
        # 別スレッドで実行
        thread = threading.Thread(target=self.optimization_worker, daemon=True)
        thread.start()
        
    def stop_optimization(self):
        self.is_running = False
        if self.process:
            self.process.terminate()
            print("\n[停止要求] プロセスを終了しました")
        
    def optimization_worker(self):
        try:
            print("=" * 80)
            print("咬合位置最適化を開始します")
            print("=" * 80)
            
            # コマンドライン引数を構築
            cmd = [
                "python",
                "app_gyu.py",
                "--upper", self.upper_stl_path.get(),
                "--lower", self.lower_stl_path.get(),
                "--move", self.move_mode.get()
            ]
            # 片顎モードの場合は追加引数
            if self.arch_mode.get() == "partial":
                cmd += ["--partial-arch", "--arch-side", self.arch_side.get()]
            
            print(f"\n実行コマンド: {' '.join(cmd)}\n")
            
            # サブプロセスで実行
            # app_gyu.pyのあるディレクトリで実行
            script_dir = str(Path(__file__).parent)
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=script_dir
            )
            
            # リアルタイム出力
            for line in self.process.stdout:
                if not self.is_running:
                    break
                print(line.rstrip())
                
            self.process.wait()
            
            if self.process.returncode == 0:
                print("\n" + "=" * 80)
                print("✓ 最適化が正常に完了しました")
                print("=" * 80)
                self.root.after(0, lambda: messagebox.showinfo("完了", 
                    "最適化が正常に完了しました"))
            else:
                print(f"\n[エラー] プロセスが異常終了しました (Exit Code: {self.process.returncode})")
                
        except Exception as e:
            print(f"\n[エラー] {str(e)}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: messagebox.showerror("エラー", 
                f"最適化中にエラーが発生しました:\n{str(e)}"))
        finally:
            self.is_running = False
            self.process = None
            self.root.after(0, self.finish_optimization)
            
    def finish_optimization(self):
        self.progress.stop()
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)


def main():
    root = tk.Tk()
    app = OcclusionOptimizerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

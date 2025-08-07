#!/usr/bin/env python3
"""
高速バッチ分析ツール

複数のraw_dataファイルを最小限の出力で高速処理
"""

import os
import glob
import shutil
import sys
import datetime
import warnings
from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')  # GUI表示を無効化して高速化
warnings.filterwarnings('ignore')  # 警告を抑制

# 既存のツールをインポート
from smoothing_test import RawDataSmoothingVisualizer
from all_isat_comparison import AllIsatComparator

class FastBatchAnalyzer:
    """高速バッチ分析クラス"""
    
    def __init__(self, raw_data_dir="raw_data", output_base_dir="results"):
        self.raw_data_dir = raw_data_dir
        self.output_base_dir = output_base_dir
        self.processed_files = []
        self.failed_files = []
        
    def find_data_files(self):
        """raw_dataディレクトリ内のデータファイルを検索"""
        patterns = ["*.txt", "*.dat", "*DivIis_tor_sum*.txt", "*DivIis_tor_sum*.dat"]
        data_files = []
        for pattern in patterns:
            search_path = os.path.join(self.raw_data_dir, pattern)
            found_files = glob.glob(search_path)
            data_files.extend(found_files)
        return list(set(data_files))
    
    def extract_shot_number(self, file_path):
        """ファイル名からshot番号を抽出"""
        filename = os.path.basename(file_path)
        patterns = [r'@(\d+)_', r'@(\d+)\.', r'_(\d+)_', r'_(\d+)\.', r'(\d{6})', r'(\d{5})']
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return match.group(1)
        return os.path.splitext(filename)[0]
    
    def create_output_directory(self, shot_number):
        """出力ディレクトリを作成"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(self.output_base_dir, f"shot_{shot_number}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def analyze_single_file(self, file_path):
        """単一ファイルの高速分析"""
        shot_number = self.extract_shot_number(file_path)
        output_dir = self.create_output_directory(shot_number)
        original_dir = os.getcwd()
        abs_file_path = os.path.abspath(file_path)
        
        try:
            os.chdir(output_dir)
            
            # 1. 全センサー比較分析（出力完全抑制）
            comparator = AllIsatComparator(abs_file_path)
            
            # データロード
            if not comparator.load_data():
                raise Exception("Failed to load data")
            
            # 運転期間検出
            comparator.operation_start_idx, comparator.operation_end_idx = comparator.detect_plasma_operation_period()
            
            # 分析実行（出力抑制）
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            
            try:
                comparator.analyze_all_columns()
                comparator.visualize_all_original_data(save_plots=True)
                comparator.visualize_smoothing_comparison(save_plots=True)
                comparator.create_summary_statistics_table()
                comparator.recommend_best_sensors()
                comparator.create_recommended_smoothing_comparison(save_plots=True)
            finally:
                sys.stdout.close()
                sys.stdout = old_stdout
            
            # 2. 上位3センサー詳細分析
            if hasattr(comparator, 'analysis_results'):
                sensor_scores = {}
                for column in comparator.isat_columns:
                    if column in comparator.analysis_results:
                        stats = comparator.analysis_results[column]['stats']
                        score = stats['max'] * 0.5 + (stats['mean'] / stats['std'] if stats['std'] > 0 else 0) * 0.3
                        sensor_scores[column] = score
                
                top_sensors = sorted(sensor_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                
                for sensor, score in top_sensors:
                    visualizer = RawDataSmoothingVisualizer(abs_file_path)
                    
                    # 出力抑制
                    old_stdout = sys.stdout
                    sys.stdout = open(os.devnull, 'w')
                    
                    try:
                        if visualizer.load_data():
                            visualizer.analyze_data_characteristics(sensor)
                            visualizer.apply_smoothing_methods(sensor)
                            visualizer.visualize_all_comparisons(save_plots=True)
                    finally:
                        sys.stdout.close()
                        sys.stdout = old_stdout
                    
                    # ファイル移動
                    sensor_dir = f"sensor_{sensor.replace('@', '_at_').replace('/', '_')}"
                    os.makedirs(sensor_dir, exist_ok=True)
                    
                    for plot_file in glob.glob("smoothing_*.png"):
                        new_name = f"{sensor}_{plot_file}"
                        shutil.move(plot_file, os.path.join(sensor_dir, new_name))
            
            self.processed_files.append({
                'file_path': file_path,
                'shot_number': shot_number,
                'output_dir': output_dir,
                'success': True
            })
            
            return True
            
        except Exception as e:
            self.failed_files.append({
                'file_path': file_path,
                'shot_number': shot_number,
                'error': str(e)
            })
            return False
            
        finally:
            os.chdir(original_dir)
    
    def run_fast_batch_analysis(self):
        """高速バッチ分析を実行"""
        print("🚀 Fast batch analysis starting...")
        
        os.makedirs(self.output_base_dir, exist_ok=True)
        data_files = self.find_data_files()
        
        if not data_files:
            print("❌ No data files found!")
            return False
        
        print(f"📊 Processing {len(data_files)} files...")
        
        # 各ファイルを高速処理
        for i, file_path in enumerate(data_files, 1):
            filename = os.path.basename(file_path)
            print(f"[{i}/{len(data_files)}] {filename}...", end=" ", flush=True)
            
            success = self.analyze_single_file(file_path)
            print("✅" if success else "❌")
        
        # 簡潔なサマリー
        success_count = len(self.processed_files)
        failed_count = len(self.failed_files)
        
        print(f"\n🎉 Completed: {success_count}✅ {failed_count}❌")
        
        if failed_count > 0:
            print("Failed files:")
            for failed in self.failed_files:
                print(f"  - {os.path.basename(failed['file_path'])}: {failed['error'][:50]}...")
        
        return True

def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fast batch analysis of multiple Isat data files')
    parser.add_argument('--raw-data-dir', default='raw_data', help='Raw data directory')
    parser.add_argument('--output-dir', default='fast_results', help='Output directory')
    parser.add_argument('--clean-output', action='store_true', help='Clean output directory')
    
    args = parser.parse_args()
    
    if args.clean_output and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    
    analyzer = FastBatchAnalyzer(args.raw_data_dir, args.output_dir)
    success = analyzer.run_fast_batch_analysis()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
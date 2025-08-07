#!/usr/bin/env python3
"""
バッチ分析ツール

raw_dataフォルダ内の複数のデータファイルを自動処理し、
それぞれの結果を個別のフォルダに格納する
"""

import os
import glob
import shutil
import sys
import datetime
from pathlib import Path
import re

# 既存のツールをインポート
from smoothing_test import RawDataSmoothingVisualizer
from all_isat_comparison import AllIsatComparator

class BatchAnalyzer:
    """バッチ分析クラス"""
    
    def __init__(self, raw_data_dir="raw_data", output_base_dir="results"):
        self.raw_data_dir = raw_data_dir
        self.output_base_dir = output_base_dir
        self.processed_files = []
        self.failed_files = []
        
    def find_data_files(self):
        """raw_dataディレクトリ内のデータファイルを検索"""
        print(f"Searching for data files in {self.raw_data_dir}...")
        
        # 一般的なファイルパターンを検索
        patterns = [
            "*.txt",
            "*.dat", 
            "*DivIis_tor_sum*.txt",
            "*DivIis_tor_sum*.dat"
        ]
        
        data_files = []
        for pattern in patterns:
            search_path = os.path.join(self.raw_data_dir, pattern)
            found_files = glob.glob(search_path)
            data_files.extend(found_files)
        
        # 重複を除去
        data_files = list(set(data_files))
        
        print(f"Found {len(data_files)} data files:")
        for file_path in data_files:
            file_size = os.path.getsize(file_path) / 1024  # KB
            print(f"  - {os.path.basename(file_path)} ({file_size:.1f} KB)")
        
        return data_files
    
    def extract_shot_number(self, file_path):
        """ファイル名からshot番号を抽出"""
        filename = os.path.basename(file_path)
        
        # 一般的なパターンでshot番号を抽出
        patterns = [
            r'@(\d+)_',  # @123456_
            r'@(\d+)\.',  # @123456.
            r'_(\d+)_',   # _123456_
            r'_(\d+)\.',  # _123456.
            r'(\d{6})',   # 6桁の数字
            r'(\d{5})',   # 5桁の数字
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return match.group(1)
        
        # パターンが見つからない場合はファイル名から拡張子を除いた部分を使用
        return os.path.splitext(filename)[0]
    
    def create_output_directory(self, shot_number):
        """出力ディレクトリを作成"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(self.output_base_dir, f"shot_{shot_number}_{timestamp}")
        
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def analyze_single_file(self, file_path, target_columns=None):
        """単一ファイルの分析を実行"""
        # Shot番号を抽出
        shot_number = self.extract_shot_number(file_path)
        
        # 出力ディレクトリを作成
        output_dir = self.create_output_directory(shot_number)
        
        # 現在のディレクトリを保存
        original_dir = os.getcwd()
        
        # ファイルパスを絶対パスに変換
        abs_file_path = os.path.abspath(file_path)
        
        try:
            # 出力ディレクトリに移動
            os.chdir(output_dir)
            
            # 1. 全センサー比較分析（出力抑制）
            comparator = AllIsatComparator(abs_file_path)
            success = comparator.run_comprehensive_analysis(verbose=False)
            
            # 2. 個別センサー詳細分析（上位3センサー）- 高速化版
            if success and hasattr(comparator, 'analysis_results'):
                # 上位センサーを取得
                sensor_scores = {}
                for column in comparator.isat_columns:
                    if column in comparator.analysis_results:
                        stats = comparator.analysis_results[column]['stats']
                        score = stats['max'] * 0.5 + (stats['mean'] / stats['std'] if stats['std'] > 0 else 0) * 0.3
                        sensor_scores[column] = score
                
                # 上位3センサーを選択
                top_sensors = sorted(sensor_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                
                for i, (sensor, score) in enumerate(top_sensors):
                    # 個別詳細分析（出力抑制）
                    visualizer = RawDataSmoothingVisualizer(abs_file_path)
                    if visualizer.load_data():
                        # 出力を抑制して高速実行
                        stats_dict, outlier_indices = visualizer.analyze_data_characteristics(sensor)
                        visualizer.apply_smoothing_methods(sensor)
                        
                        # センサー固有のサブディレクトリを作成
                        sensor_dir = f"sensor_{sensor.replace('@', '_at_').replace('/', '_')}"
                        os.makedirs(sensor_dir, exist_ok=True)
                        
                        # 可視化を実行（出力抑制）
                        visualizer.visualize_all_comparisons(save_plots=True)
                        
                        # 生成されたファイルをセンサーディレクトリに移動
                        for plot_file in glob.glob("smoothing_*.png"):
                            new_name = f"{sensor}_{plot_file}"
                            shutil.move(plot_file, os.path.join(sensor_dir, new_name))
            
            # 3. サマリーレポートを作成
            try:
                self.create_summary_report(output_dir, shot_number, abs_file_path, comparator if success else None)
            except Exception as e:
                print(f"Warning: Failed to create summary report: {e}")
            
            self.processed_files.append({
                'file_path': file_path,
                'shot_number': shot_number,
                'output_dir': output_dir,
                'success': success
            })
            
            # 簡潔な完了通知
            print(f"✅ Shot {shot_number} completed -> {os.path.basename(output_dir)}")
            
        except Exception as e:
            print(f"❌ Error analyzing {file_path}: {e}")
            self.failed_files.append({
                'file_path': file_path,
                'shot_number': shot_number,
                'error': str(e)
            })
            
        finally:
            # 元のディレクトリに戻る
            os.chdir(original_dir)
    
    def create_summary_report(self, output_dir, shot_number, file_path, comparator=None):
        """サマリーレポートを作成"""
        report_path = os.path.join(output_dir, "analysis_summary.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Shot {shot_number} Analysis Summary\n\n")
            f.write(f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Source File:** `{os.path.basename(file_path)}`\n\n")
            
            if comparator and hasattr(comparator, 'operation_start_idx'):
                # プラズマ運転期間情報
                start_time = comparator.data['Time'].iloc[comparator.operation_start_idx]
                end_time = comparator.data['Time'].iloc[comparator.operation_end_idx]
                duration = end_time - start_time
                
                f.write(f"## 🚀 Plasma Operation Period\n\n")
                f.write(f"- **Start Time:** {start_time:.3f} s\n")
                f.write(f"- **End Time:** {end_time:.3f} s\n")
                f.write(f"- **Duration:** {duration:.3f} s\n")
                f.write(f"- **Data Points:** {comparator.operation_end_idx - comparator.operation_start_idx + 1}\n\n")
            
            if comparator and hasattr(comparator, 'analysis_results'):
                # センサーランキング
                f.write(f"## 📊 Sensor Ranking\n\n")
                f.write("| Rank | Sensor | Mean | Max | Std Dev | Outliers |\n")
                f.write("|------|--------|------|-----|---------|----------|\n")
                
                # センサーをスコア順にソート
                sensor_scores = {}
                for column in comparator.isat_columns:
                    if column in comparator.analysis_results:
                        stats = comparator.analysis_results[column]['stats']
                        score = stats['max'] * 0.5 + (stats['mean'] / stats['std'] if stats['std'] > 0 else 0) * 0.3
                        sensor_scores[column] = (score, stats)
                
                sorted_sensors = sorted(sensor_scores.items(), key=lambda x: x[1][0], reverse=True)
                
                for i, (sensor, (score, stats)) in enumerate(sorted_sensors[:10]):
                    rank_emoji = ["🥇", "🥈", "🥉"] if i < 3 else ["📍"]
                    emoji = rank_emoji[0] if i < 3 else rank_emoji[0]
                    
                    f.write(f"| {emoji} {i+1} | **{sensor}** | {stats['mean']:.6f} | "
                           f"{stats['max']:.6f} | {stats['std']:.6f} | {stats['outlier_count']} |\n")
                
                # 推奨事項
                f.write(f"\n## 🎯 Recommendations\n\n")
                if sorted_sensors:
                    best_sensor = sorted_sensors[0][0]
                    f.write(f"- **Primary sensor:** `{best_sensor}` (highest overall score)\n")
                    if len(sorted_sensors) >= 3:
                        top_3 = [sensor for sensor, _ in sorted_sensors[:3]]
                        f.write(f"- **Top 3 sensors:** {', '.join([f'`{s}`' for s in top_3])}\n")
                
                f.write(f"- **Recommended smoothing:** Adaptive Gaussian (σ=1.5 base, σ=3.0 outliers)\n")
            
            # ファイルリスト
            f.write(f"\n## 📁 Generated Files\n\n")
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file.endswith(('.png', '.csv', '.txt')):
                        rel_path = os.path.relpath(os.path.join(root, file), output_dir)
                        f.write(f"- `{rel_path}`\n")
    
    def run_batch_analysis(self, target_columns=None):
        """バッチ分析を実行"""
        print("🚀 Starting batch analysis...")
        print(f"📁 Raw data directory: {self.raw_data_dir}")
        print(f"📁 Output base directory: {self.output_base_dir}")
        
        # 出力ベースディレクトリを作成
        os.makedirs(self.output_base_dir, exist_ok=True)
        
        # データファイルを検索
        data_files = self.find_data_files()
        
        if not data_files:
            print("❌ No data files found!")
            return False
        
        print(f"\n📊 Processing {len(data_files)} files...")
        
        # 各ファイルを処理（簡潔な進捗表示）
        for i, file_path in enumerate(data_files, 1):
            print(f"[{i}/{len(data_files)}] {os.path.basename(file_path)}")
            self.analyze_single_file(file_path, target_columns)
        
        # 総括レポートを作成
        self.create_batch_summary()
        
        print(f"\n🎉 Batch analysis completed!")
        print(f"✅ Successfully processed: {len(self.processed_files)} files")
        print(f"❌ Failed: {len(self.failed_files)} files")
        
        return True
    
    def create_batch_summary(self):
        """バッチ処理の総括レポートを作成"""
        summary_path = os.path.join(self.output_base_dir, "batch_summary.md")
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("# Batch Analysis Summary\n\n")
            f.write(f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"## 📊 Processing Results\n\n")
            f.write(f"- **Total files:** {len(self.processed_files) + len(self.failed_files)}\n")
            f.write(f"- **Successfully processed:** {len(self.processed_files)}\n")
            f.write(f"- **Failed:** {len(self.failed_files)}\n\n")
            
            if self.processed_files:
                f.write(f"## ✅ Successfully Processed Files\n\n")
                f.write("| Shot | File | Output Directory |\n")
                f.write("|------|------|------------------|\n")
                
                for result in self.processed_files:
                    filename = os.path.basename(result['file_path'])
                    output_rel = os.path.relpath(result['output_dir'], self.output_base_dir)
                    f.write(f"| {result['shot_number']} | `{filename}` | `{output_rel}` |\n")
            
            if self.failed_files:
                f.write(f"\n## ❌ Failed Files\n\n")
                f.write("| Shot | File | Error |\n")
                f.write("|------|------|-------|\n")
                
                for result in self.failed_files:
                    filename = os.path.basename(result['file_path'])
                    error = result['error'][:100] + "..." if len(result['error']) > 100 else result['error']
                    f.write(f"| {result['shot_number']} | `{filename}` | {error} |\n")
        
        print(f"📋 Batch summary saved: {summary_path}")

def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch analysis of multiple Isat data files')
    parser.add_argument('--raw-data-dir', default='raw_data',
                       help='Directory containing raw data files (default: raw_data)')
    parser.add_argument('--output-dir', default='results',
                       help='Base directory for output results (default: results)')
    parser.add_argument('--columns', nargs='*',
                       help='Specific columns to analyze (default: all available)')
    parser.add_argument('--clean-output', action='store_true',
                       help='Clean output directory before starting')
    
    args = parser.parse_args()
    
    # 出力ディレクトリのクリーンアップ
    if args.clean_output and os.path.exists(args.output_dir):
        print(f"🧹 Cleaning output directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    
    # バッチ分析を実行
    analyzer = BatchAnalyzer(args.raw_data_dir, args.output_dir)
    success = analyzer.run_batch_analysis(args.columns)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
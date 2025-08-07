#!/usr/bin/env python3
"""
全Isatカラムの比較ツール

全てのIsatセンサーのデータを可視化し、
最適なスムージング手法を適用した結果を比較する
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import sys
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

class AllIsatComparator:
    """全Isatカラムの比較クラス"""
    
    def __init__(self, data_file_path):
        self.data_file_path = data_file_path
        self.data = None
        self.isat_columns = []
        self.analysis_results = {}
        
    def load_data(self):
        """データを読み込む"""
        print(f"Loading data from: {self.data_file_path}")
        
        if not os.path.exists(self.data_file_path):
            print(f"Error: File not found: {self.data_file_path}")
            return False
        
        try:
            # コメント行をスキップしてデータの開始位置を特定
            with open(self.data_file_path, "r") as file:
                lines = file.readlines()
            
            # データセクションの開始を見つける
            data_start = next(i for i, line in enumerate(lines) 
                            if not line.startswith('#') and re.match(r'\s*\d', line))
            
            # データを読み込み
            self.data = pd.read_csv(self.data_file_path, skiprows=data_start, header=None)
            
            # カラム名を設定
            column_names = ['Time', 'Iis_2L@20', 'Iis_2R@20', 'Iis_4L@20', 'Iis_4R@19',
                           'Iis_6L@20', 'Iis_6R@4', 'Iis_7L@20', 'Iis_7R@19',
                           'Iis_8L@20', 'Iis_8R@20', 'Iis_9L@17', 'Iis_9R@19',
                           'Iis_10L@20', 'Iis_10R@20']
            
            self.data.columns = column_names
            self.isat_columns = [col for col in column_names if col.startswith('Iis_')]
            
            print(f"Successfully loaded data: {len(self.data)} points")
            print(f"Available Isat columns: {len(self.isat_columns)} sensors")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def detect_plasma_operation_period(self):
        """プラズマ運転期間を自動検出"""
        print("Detecting plasma operation period...")
        
        # 全Isatセンサーの合計値を使用して活動期間を検出
        total_activity = np.zeros(len(self.data))
        for col in self.isat_columns:
            total_activity += np.abs(self.data[col].values)
        
        # 活動レベルの閾値を設定（最大値の5%）
        activity_threshold = np.max(total_activity) * 0.05
        
        # 活動期間を検出
        active_indices = np.where(total_activity > activity_threshold)[0]
        
        if len(active_indices) == 0:
            print("Warning: No significant plasma activity detected. Using full time range.")
            return 0, len(self.data) - 1
        
        # 連続する活動期間を検出
        start_idx = active_indices[0]
        end_idx = active_indices[-1]
        
        # 前後にマージンを追加（データの5%程度）
        margin = int(0.05 * len(self.data))
        start_idx = max(0, start_idx - margin)
        end_idx = min(len(self.data) - 1, end_idx + margin)
        
        operation_start_time = self.data['Time'].iloc[start_idx]
        operation_end_time = self.data['Time'].iloc[end_idx]
        total_time = self.data['Time'].iloc[-1] - self.data['Time'].iloc[0]
        operation_duration = operation_end_time - operation_start_time
        
        print(f"  Plasma operation detected:")
        print(f"    Start time: {operation_start_time:.3f} s (index: {start_idx})")
        print(f"    End time: {operation_end_time:.3f} s (index: {end_idx})")
        print(f"    Duration: {operation_duration:.3f} s")
        print(f"    Operation ratio: {operation_duration/total_time:.1%} of total time")
        
        return start_idx, end_idx

    def analyze_all_columns(self):
        """全てのIsatカラムを分析"""
        print("\nAnalyzing all Isat columns...")
        
        # プラズマ運転期間を検出
        self.operation_start_idx, self.operation_end_idx = self.detect_plasma_operation_period()
        
        for column in self.isat_columns:
            print(f"Analyzing {column}...")
            
            # 運転期間のデータのみを使用
            data_values = self.data[column].iloc[self.operation_start_idx:self.operation_end_idx+1].values
            
            # 基本統計
            stats = {
                'mean': np.mean(data_values),
                'median': np.median(data_values),
                'std': np.std(data_values),
                'min': np.min(data_values),
                'max': np.max(data_values),
                'range': np.max(data_values) - np.min(data_values)
            }
            
            # 外れ値検出
            window_size = 10
            rolling_median = np.array([
                np.median(data_values[max(0, i-window_size//2):min(len(data_values), i+window_size//2+1)])
                for i in range(len(data_values))
            ])
            
            deviation = data_values - rolling_median
            threshold = -2.5 * np.std(deviation)
            outlier_indices = np.where(deviation < threshold)[0]
            
            stats['outlier_count'] = len(outlier_indices)
            stats['outlier_ratio'] = len(outlier_indices) / len(data_values)
            
            # 推奨スムージング手法を適用
            recommended_methods = self._apply_recommended_smoothing(data_values, stats)
            
            self.analysis_results[column] = {
                'stats': stats,
                'outlier_indices': outlier_indices,
                'original_data': data_values,
                'smoothed_data': recommended_methods
            }
        
        print("Analysis completed for all columns.")
        return True
    
    def _apply_recommended_smoothing(self, data, stats):
        """推奨スムージング手法を適用"""
        results = {}
        
        # 1. Gaussian (中程度) - バランス重視
        results['gaussian_medium'] = {
            'data': gaussian_filter1d(data, sigma=1.5),
            'name': 'Gaussian (σ=1.5)',
            'description': 'Balanced noise reduction and signal preservation'
        }
        
        # 2. Savitzky-Golay - エッジ保持
        try:
            results['savgol'] = {
                'data': savgol_filter(data, window_length=7, polyorder=2),
                'name': 'Savitzky-Golay (w=7, p=2)',
                'description': 'Edge-preserving smoothing'
            }
        except:
            results['savgol'] = results['gaussian_medium'].copy()
            results['savgol']['name'] = 'Savitzky-Golay (fallback)'
        
        # 3. 適応的スムージング - センサー特性考慮
        results['adaptive'] = {
            'data': self._apply_adaptive_smoothing_simple(data, stats),
            'name': 'Adaptive Smoothing',
            'description': 'Sensor characteristic-aware smoothing'
        }
        
        return results
    
    def _apply_adaptive_smoothing_simple(self, data, stats):
        """簡単な適応的スムージング"""
        # 基本のGaussianスムージング
        smoothed = gaussian_filter1d(data, sigma=1.5)
        
        # 極端に低い値の領域により強いスムージング
        low_threshold = stats['min'] + 0.1 * stats['range']
        low_value_mask = data < low_threshold
        
        if np.any(low_value_mask):
            # 低い値の領域に追加スムージング
            local_data = data.copy()
            local_data[low_value_mask] = gaussian_filter1d(local_data[low_value_mask], sigma=3.0)
            
            # 境界をスムーズに
            smoothed = 0.7 * smoothed + 0.3 * local_data
        
        return smoothed
    
    def visualize_all_original_data(self, save_plots=True):
        """全ての原データを可視化"""
        print("Creating original data comparison...")
        
        # 運転期間のデータのみを使用
        time_data = self.data['Time'].iloc[self.operation_start_idx:self.operation_end_idx+1].values
        
        # 全てのIsatカラムを一つのプロットに
        fig, axes = plt.subplots(3, 5, figsize=(20, 15))
        axes = axes.flatten()
        
        for i, column in enumerate(self.isat_columns):
            if i < len(axes):
                ax = axes[i]
                data_values = self.data[column].iloc[self.operation_start_idx:self.operation_end_idx+1].values
                
                ax.plot(time_data, data_values, 'b-', alpha=0.7, linewidth=1)
                ax.set_title(f'{column}', fontsize=10)
                ax.set_xlabel('Time (s)', fontsize=8)
                ax.set_ylabel('Isat (A)', fontsize=8)
                ax.grid(True, alpha=0.3)
                
                # 統計情報を表示
                mean_val = np.mean(data_values)
                max_val = np.max(data_values)
                ax.text(0.02, 0.95, f'Max: {max_val:.4f}\nMean: {mean_val:.4f}', 
                       transform=ax.transAxes, fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 余ったサブプロットを非表示
        for i in range(len(self.isat_columns), len(axes)):
            axes[i].set_visible(False)
        
        operation_time = self.data['Time'].iloc[self.operation_start_idx:self.operation_end_idx+1]
        plt.suptitle(f'All Isat Sensors - Original Data (Operation Period: {operation_time.iloc[0]:.2f}-{operation_time.iloc[-1]:.2f}s)', fontsize=16)
        plt.tight_layout()
        
        if save_plots:
            filename = 'all_isat_original_comparison.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Original data comparison saved as: {filename}")
        
        plt.show()
    
    def visualize_smoothing_comparison(self, save_plots=True):
        """スムージング比較を可視化"""
        print("Creating smoothing comparison for all sensors...")
        
        # 運転期間のデータのみを使用
        time_data = self.data['Time'].iloc[self.operation_start_idx:self.operation_end_idx+1].values
        
        # 各センサーについて3つの手法を比較
        n_sensors = len(self.isat_columns)
        fig, axes = plt.subplots(n_sensors, 1, figsize=(16, 3*n_sensors))
        
        if n_sensors == 1:
            axes = [axes]
        
        for i, column in enumerate(self.isat_columns):
            ax = axes[i]
            
            # 原データ
            original_data = self.analysis_results[column]['original_data']
            ax.plot(time_data, original_data, 'lightblue', alpha=0.6, linewidth=1, label='Original')
            
            # スムージング結果
            smoothed_data = self.analysis_results[column]['smoothed_data']
            colors = ['red', 'green', 'purple']
            methods = ['gaussian_medium', 'savgol', 'adaptive']
            
            for method, color in zip(methods, colors):
                if method in smoothed_data:
                    ax.plot(time_data, smoothed_data[method]['data'], 
                           color=color, linewidth=2, alpha=0.8,
                           label=smoothed_data[method]['name'])
            
            ax.set_title(f'{column} - Smoothing Comparison')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Isat (A)')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            filename = 'all_isat_smoothing_comparison.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Smoothing comparison saved as: {filename}")
        
        plt.show()
    
    def create_summary_statistics_table(self):
        """統計サマリーテーブルを作成"""
        print("\n" + "="*120)
        print("ALL ISAT SENSORS SUMMARY STATISTICS")
        print("="*120)
        print(f"{'Sensor':<12} {'Mean':<10} {'Median':<10} {'Std Dev':<10} {'Min':<10} {'Max':<10} {'Range':<10} {'Outliers':<10}")
        print("-"*120)
        
        for column in self.isat_columns:
            stats = self.analysis_results[column]['stats']
            outlier_count = stats['outlier_count']
            
            print(f"{column:<12} {stats['mean']:<10.5f} {stats['median']:<10.5f} "
                  f"{stats['std']:<10.5f} {stats['min']:<10.5f} {stats['max']:<10.5f} "
                  f"{stats['range']:<10.5f} {outlier_count:<10d}")
        
        print("="*120)
    
    def recommend_best_sensors(self):
        """最適なセンサーを推奨"""
        print("\nSENSOR RECOMMENDATIONS:")
        print("="*60)
        
        # 各センサーをスコア付け
        sensor_scores = {}
        
        for column in self.isat_columns:
            stats = self.analysis_results[column]['stats']
            
            # スコア計算（高い値ほど良い）
            # 1. 信号強度（最大値）
            signal_strength = stats['max']
            
            # 2. 信号対ノイズ比（平均/標準偏差）
            snr = stats['mean'] / stats['std'] if stats['std'] > 0 else 0
            
            # 3. 外れ値の少なさ（逆数）
            outlier_penalty = 1 / (1 + stats['outlier_ratio'])
            
            # 4. ダイナミックレンジ
            dynamic_range = stats['range']
            
            # 総合スコア（重み付き平均）
            total_score = (
                0.3 * signal_strength +
                0.3 * snr +
                0.2 * outlier_penalty +
                0.2 * dynamic_range
            )
            
            sensor_scores[column] = {
                'total_score': total_score,
                'signal_strength': signal_strength,
                'snr': snr,
                'outlier_penalty': outlier_penalty,
                'dynamic_range': dynamic_range
            }
        
        # スコア順にソート
        sorted_sensors = sorted(sensor_scores.items(), key=lambda x: x[1]['total_score'], reverse=True)
        
        print("Ranking (Total Score | Signal Strength | SNR | Outlier Penalty | Dynamic Range):")
        print("-"*90)
        
        for i, (sensor, scores) in enumerate(sorted_sensors):
            print(f"{i+1:2d}. {sensor:<12} | {scores['total_score']:.4f} | "
                  f"{scores['signal_strength']:.4f} | {scores['snr']:.4f} | "
                  f"{scores['outlier_penalty']:.4f} | {scores['dynamic_range']:.4f}")
        
        # 推奨
        best_sensor = sorted_sensors[0][0]
        top_3 = [sensor for sensor, _ in sorted_sensors[:3]]
        
        print("\nRECOMMENDATIONS:")
        print(f"  Best overall sensor: {best_sensor}")
        print(f"  Top 3 sensors: {', '.join(top_3)}")
        print(f"  Recommended for analysis: {best_sensor} (highest signal quality)")
        
        return sorted_sensors
    
    def create_recommended_smoothing_comparison(self, top_sensors=3, save_plots=True):
        """推奨センサーのスムージング比較"""
        # 上位センサーを取得
        sorted_sensors = self.recommend_best_sensors()
        top_sensor_names = [sensor for sensor, _ in sorted_sensors[:top_sensors]]
        
        print(f"\nCreating detailed comparison for top {top_sensors} sensors...")
        
        # 運転期間のデータのみを使用
        time_data = self.data['Time'].iloc[self.operation_start_idx:self.operation_end_idx+1].values
        
        fig, axes = plt.subplots(top_sensors, 1, figsize=(16, 4*top_sensors))
        if top_sensors == 1:
            axes = [axes]
        
        for i, sensor in enumerate(top_sensor_names):
            ax = axes[i]
            
            # 原データ
            original_data = self.analysis_results[sensor]['original_data']
            ax.plot(time_data, original_data, 'lightblue', alpha=0.5, linewidth=1, label='Original')
            
            # 推奨スムージング（適応的）
            adaptive_data = self.analysis_results[sensor]['smoothed_data']['adaptive']['data']
            ax.plot(time_data, adaptive_data, 'red', linewidth=2, alpha=0.9, label='Recommended: Adaptive Smoothing')
            
            # Gaussianスムージング（比較用）
            gaussian_data = self.analysis_results[sensor]['smoothed_data']['gaussian_medium']['data']
            ax.plot(time_data, gaussian_data, 'green', linewidth=2, alpha=0.7, label='Gaussian (σ=1.5)')
            
            # 統計情報
            stats = self.analysis_results[sensor]['stats']
            ax.text(0.02, 0.95, 
                   f'Max: {stats["max"]:.4f}\nMean: {stats["mean"]:.4f}\nOutliers: {stats["outlier_count"]}', 
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_title(f'{sensor} - Recommended Smoothing (Rank #{i+1})')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Isat (A)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            filename = 'top_sensors_recommended_smoothing.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Top sensors comparison saved as: {filename}")
        
        plt.show()
        
        return top_sensor_names
    
    def run_comprehensive_analysis(self):
        """包括的な分析を実行"""
        print("Starting comprehensive analysis of all Isat sensors...")
        print("="*70)
        
        # データ読み込み
        if not self.load_data():
            return False
        
        # 全カラム分析
        if not self.analyze_all_columns():
            return False
        
        # 可視化
        self.visualize_all_original_data()
        self.visualize_smoothing_comparison()
        
        # 統計サマリー
        self.create_summary_statistics_table()
        
        # 推奨
        top_sensors = self.create_recommended_smoothing_comparison()
        
        print("\nAnalysis completed successfully!")
        print("\nSUMMARY:")
        print("- All Isat sensors have been analyzed and compared")
        print("- Recommended smoothing methods have been applied")
        print("- Top sensors have been identified based on signal quality")
        print("- Use the adaptive smoothing method for best results")
        
        return top_sensors

def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare all Isat sensors and their smoothing options')
    parser.add_argument('--data-file', 
                       default='/Users/maesawayuta/yamada_lab/fusion-plasma-ml-toolkit/test/raw_data/DivIis_tor_sum@163402_1.txt',
                       help='Path to the raw data file')
    parser.add_argument('--top-sensors', type=int, default=3,
                       help='Number of top sensors to analyze in detail (default: 3)')
    
    args = parser.parse_args()
    
    # 分析実行
    comparator = AllIsatComparator(args.data_file)
    top_sensors = comparator.run_comprehensive_analysis()
    
    if not top_sensors:
        sys.exit(1)

if __name__ == "__main__":
    main()
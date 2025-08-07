#!/usr/bin/env python3
"""
生データのスムージング比較可視化ツール

raw_dataフォルダ内の時系列データを読み込み、
様々なスムージング手法を適用して比較可視化を行う
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import sys
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy import stats

class RawDataSmoothingVisualizer:
    """生データのスムージング比較可視化クラス"""
    
    def __init__(self, data_file_path):
        self.data_file_path = data_file_path
        self.data = None
        self.target_columns = ['Iis_7L@20']  # メインターゲット（後で拡張可能）
        self.all_isat_columns = []  # 全Isatカラム
        self.smoothed_results = {}
        
    def load_data(self):
        """生データを読み込む"""
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
            
            # カラム名を設定（メタデータから抽出）
            column_names = ['Time', 'Iis_2L@20', 'Iis_2R@20', 'Iis_4L@20', 'Iis_4R@19',
                           'Iis_6L@20', 'Iis_6R@4', 'Iis_7L@20', 'Iis_7R@19',
                           'Iis_8L@20', 'Iis_8R@20', 'Iis_9L@17', 'Iis_9R@19',
                           'Iis_10L@20', 'Iis_10R@20']
            
            self.data.columns = column_names
            self.all_isat_columns = [col for col in column_names if col.startswith('Iis_')]
            
            print(f"Successfully loaded data: {len(self.data)} points")
            print(f"Available Isat columns: {self.all_isat_columns}")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def detect_plasma_operation_period(self):
        """プラズマ運転期間を自動検出"""
        print("Detecting plasma operation period...")
        
        # 全Isatセンサーの合計値を使用して活動期間を検出
        isat_columns = [col for col in self.data.columns if col.startswith('Iis_')]
        
        # 各時刻での全センサーの絶対値の合計を計算
        total_activity = np.zeros(len(self.data))
        for col in isat_columns:
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

    def analyze_data_characteristics(self, column='Iis_7L@20'):
        """データの特性を分析"""
        if self.data is None or column not in self.data.columns:
            print(f"Error: Data not loaded or column {column} not found")
            return None
        
        # プラズマ運転期間を検出
        start_idx, end_idx = self.detect_plasma_operation_period()
        
        # 運転期間のデータのみを使用
        values = self.data[column].iloc[start_idx:end_idx+1].values
        time_range = self.data['Time'].iloc[start_idx:end_idx+1]
        
        # 基本統計
        stats_dict = {
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'percentiles': {p: np.percentile(values, p) for p in [1, 5, 10, 25, 75, 90, 95, 99]},
            'operation_start_idx': start_idx,
            'operation_end_idx': end_idx,
            'operation_time_range': (time_range.iloc[0], time_range.iloc[-1])
        }
        
        # 外れ値検出（移動中央値ベース）
        window_size = 10
        rolling_median = np.array([
            np.median(values[max(0, i-window_size//2):min(len(values), i+window_size//2+1)])
            for i in range(len(values))
        ])
        
        deviation = values - rolling_median
        threshold = -2.5 * np.std(deviation)  # 負の閾値（低い値を検出）
        outlier_indices = np.where(deviation < threshold)[0]
        
        stats_dict['outlier_count'] = len(outlier_indices)
        stats_dict['outlier_ratio'] = len(outlier_indices) / len(values)
        
        print(f"\nData characteristics for {column} (operation period only):")
        print(f"  Data points: {len(values)} (operation period)")
        print(f"  Time range: {stats_dict['operation_time_range'][0]:.3f} - {stats_dict['operation_time_range'][1]:.3f} s")
        print(f"  Mean: {stats_dict['mean']:.6f}")
        print(f"  Median: {stats_dict['median']:.6f}")
        print(f"  Std Dev: {stats_dict['std']:.6f}")
        print(f"  Range: [{stats_dict['min']:.6f}, {stats_dict['max']:.6f}]")
        print(f"  Outliers detected: {stats_dict['outlier_count']} ({stats_dict['outlier_ratio']:.2%})")
        
        return stats_dict, outlier_indices
    
    def apply_smoothing_methods(self, column='Iis_7L@20'):
        """様々なスムージング手法を適用"""
        if self.data is None or column not in self.data.columns:
            print(f"Error: Data not loaded or column {column} not found")
            return False
        
        # プラズマ運転期間を検出
        start_idx, end_idx = self.detect_plasma_operation_period()
        
        # 運転期間のデータのみを使用
        original_data = self.data[column].iloc[start_idx:end_idx+1].values
        time_data = self.data['Time'].iloc[start_idx:end_idx+1].values
        
        # 運転期間情報を保存
        self.operation_period = {
            'start_idx': start_idx,
            'end_idx': end_idx,
            'time_range': (time_data[0], time_data[-1])
        }
        
        print(f"\nApplying various smoothing methods to {column} (operation period only)...")
        print(f"Time range: {time_data[0]:.3f} - {time_data[-1]:.3f} s ({len(time_data)} points)")
        
        # スムージング手法とパラメータの定義
        smoothing_methods = {
            # Gaussianフィルタ
            'gaussian_light': {'method': 'gaussian', 'sigma': 0.5, 'name': 'Gaussian (σ=0.5)'},
            'gaussian_medium': {'method': 'gaussian', 'sigma': 1.0, 'name': 'Gaussian (σ=1.0)'},
            'gaussian_strong': {'method': 'gaussian', 'sigma': 2.0, 'name': 'Gaussian (σ=2.0)'},
            'gaussian_very_strong': {'method': 'gaussian', 'sigma': 3.0, 'name': 'Gaussian (σ=3.0)'},
            
            # Savitzky-Golayフィルタ
            'savgol_5_2': {'method': 'savgol', 'window': 5, 'poly': 2, 'name': 'Savitzky-Golay (w=5, p=2)'},
            'savgol_7_2': {'method': 'savgol', 'window': 7, 'poly': 2, 'name': 'Savitzky-Golay (w=7, p=2)'},
            'savgol_11_2': {'method': 'savgol', 'window': 11, 'poly': 2, 'name': 'Savitzky-Golay (w=11, p=2)'},
            'savgol_15_3': {'method': 'savgol', 'window': 15, 'poly': 3, 'name': 'Savitzky-Golay (w=15, p=3)'},
            
            # 移動平均
            'moving_avg_3': {'method': 'moving_avg', 'window': 3, 'name': 'Moving Average (w=3)'},
            'moving_avg_5': {'method': 'moving_avg', 'window': 5, 'name': 'Moving Average (w=5)'},
            'moving_avg_7': {'method': 'moving_avg', 'window': 7, 'name': 'Moving Average (w=7)'},
            'moving_avg_11': {'method': 'moving_avg', 'window': 11, 'name': 'Moving Average (w=11)'},
        }
        
        # 適応的スムージング（センサー特性考慮）
        stats_dict, outlier_indices = self.analyze_data_characteristics(column)
        adaptive_methods = self._create_adaptive_methods(stats_dict, outlier_indices)
        smoothing_methods.update(adaptive_methods)
        
        # 各手法を適用
        results = {'original': {'data': original_data, 'name': 'Original'}}
        
        for key, params in smoothing_methods.items():
            try:
                if params['method'] == 'gaussian':
                    smoothed = gaussian_filter1d(original_data, sigma=params['sigma'])
                    
                elif params['method'] == 'savgol':
                    window = params['window']
                    poly = params['poly']
                    if window >= len(original_data):
                        window = len(original_data) - 1
                        if window % 2 == 0:
                            window -= 1
                    if poly >= window:
                        poly = window - 1
                    smoothed = savgol_filter(original_data, window, poly)
                    
                elif params['method'] == 'moving_avg':
                    window = params['window']
                    kernel = np.ones(window) / window
                    smoothed = np.convolve(original_data, kernel, mode='same')
                    
                elif params['method'] == 'adaptive':
                    smoothed = self._apply_adaptive_smoothing(
                        original_data, outlier_indices, params
                    )
                
                results[key] = {
                    'data': smoothed,
                    'name': params['name'],
                    'params': params
                }
                print(f"  ✓ {params['name']}")
                
            except Exception as e:
                print(f"  ✗ {params['name']}: {e}")
        
        self.smoothed_results = results
        return True
    
    def _create_adaptive_methods(self, stats_dict, outlier_indices):
        """適応的スムージング手法を作成"""
        adaptive_methods = {
            'adaptive_light': {
                'method': 'adaptive',
                'base_sigma': 1.0,
                'outlier_sigma': 2.0,
                'name': 'Adaptive Light (base=1.0, outlier=2.0)'
            },
            'adaptive_medium': {
                'method': 'adaptive',
                'base_sigma': 1.5,
                'outlier_sigma': 3.0,
                'name': 'Adaptive Medium (base=1.5, outlier=3.0)'
            },
            'adaptive_strong': {
                'method': 'adaptive',
                'base_sigma': 2.0,
                'outlier_sigma': 4.0,
                'name': 'Adaptive Strong (base=2.0, outlier=4.0)'
            }
        }
        return adaptive_methods
    
    def _apply_adaptive_smoothing(self, data, outlier_indices, params):
        """適応的スムージングを適用"""
        smoothed_data = data.copy()
        base_sigma = params['base_sigma']
        outlier_sigma = params['outlier_sigma']
        
        # 値に基づく重み付け
        low_threshold = np.percentile(data, 10)
        high_threshold = np.percentile(data, 75)
        
        # 全体的に軽いスムージング
        smoothed_data = gaussian_filter1d(smoothed_data, sigma=base_sigma)
        
        # 外れ値領域に追加の強いスムージング
        if len(outlier_indices) > 0:
            # 連続する外れ値領域をグループ化
            groups = []
            current_group = [outlier_indices[0]]
            
            for i in range(1, len(outlier_indices)):
                if outlier_indices[i] - outlier_indices[i-1] <= 5:  # 5ポイント以内
                    current_group.append(outlier_indices[i])
                else:
                    groups.append(current_group)
                    current_group = [outlier_indices[i]]
            groups.append(current_group)
            
            # 各グループに強いスムージング
            for group in groups:
                if len(group) < 3:
                    continue
                    
                start_idx = max(0, group[0] - 10)
                end_idx = min(len(data), group[-1] + 11)
                
                local_data = smoothed_data[start_idx:end_idx]
                local_smoothed = gaussian_filter1d(local_data, sigma=outlier_sigma)
                
                # 対象領域のみ更新
                smoothed_data[start_idx:end_idx] = local_smoothed
        
        return smoothed_data
    
    def calculate_metrics(self):
        """各スムージング手法の評価メトリクスを計算"""
        if not self.smoothed_results:
            print("Error: No smoothing results available")
            return None
        
        original_data = self.smoothed_results['original']['data']
        original_gradient = np.gradient(original_data)
        original_noise_level = np.std(original_gradient)
        
        metrics = {}
        
        for key, result in self.smoothed_results.items():
            if key == 'original':
                continue
                
            smoothed_data = result['data']
            
            # ノイズ削減効果
            smoothed_gradient = np.gradient(smoothed_data)
            noise_level = np.std(smoothed_gradient)
            noise_reduction = (original_noise_level - noise_level) / original_noise_level * 100
            
            # 元データとの相関
            correlation = np.corrcoef(original_data, smoothed_data)[0, 1]
            
            # 平滑度（二階微分の標準偏差）
            second_derivative = np.gradient(smoothed_gradient)
            smoothness = np.std(second_derivative)
            
            # 信号保持率（高い値の保持度）
            high_value_mask = original_data > np.percentile(original_data, 75)
            if np.any(high_value_mask):
                high_value_preservation = np.corrcoef(
                    original_data[high_value_mask], 
                    smoothed_data[high_value_mask]
                )[0, 1]
            else:
                high_value_preservation = 1.0
            
            metrics[key] = {
                'noise_reduction': noise_reduction,
                'correlation': correlation,
                'smoothness': smoothness,
                'high_value_preservation': high_value_preservation,
                'name': result['name']
            }
        
        return metrics
    
    def visualize_all_comparisons(self, save_plots=True):
        """全ての比較結果を可視化"""
        if not self.smoothed_results:
            print("Error: No smoothing results available")
            return
        
        print("Creating comprehensive comparison visualizations...")
        
        # 1. 全体比較プロット
        self._plot_overall_comparison(save_plots)
        
        # 2. カテゴリ別比較プロット
        self._plot_category_comparisons(save_plots)
        
        # 3. メトリクス比較プロット
        self._plot_metrics_comparison(save_plots)
        
        # 4. 推奨手法の詳細比較
        self._plot_recommended_methods(save_plots)
    
    def _plot_overall_comparison(self, save_plots):
        """全体比較プロット"""
        # 運転期間のデータを使用
        if hasattr(self, 'operation_period'):
            start_idx = self.operation_period['start_idx']
            end_idx = self.operation_period['end_idx']
            time_data = self.data['Time'].iloc[start_idx:end_idx+1].values
        else:
            time_data = self.data['Time'].values
        
        original_data = self.smoothed_results['original']['data']
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # すべての手法をプロット（透明度調整）
        methods_to_plot = ['gaussian_medium', 'savgol_7_2', 'moving_avg_5', 'adaptive_medium']
        colors = ['red', 'green', 'orange', 'purple']
        
        for i, (ax, (method_key, color)) in enumerate(zip(axes.flat, zip(methods_to_plot, colors))):
            ax.plot(time_data, original_data, 'b-', alpha=0.5, linewidth=1, label='Original')
            
            if method_key in self.smoothed_results:
                smoothed_data = self.smoothed_results[method_key]['data']
                method_name = self.smoothed_results[method_key]['name']
                ax.plot(time_data, smoothed_data, color=color, linewidth=2, label=method_name)
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Isat (A)')
            ax.set_title(f'Comparison: {self.smoothed_results[method_key]["name"] if method_key in self.smoothed_results else method_key}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            filename = 'smoothing_overall_comparison.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Overall comparison saved as: {filename}")
        
        plt.show()
    
    def _plot_category_comparisons(self, save_plots):
        """カテゴリ別比較プロット"""
        # 運転期間のデータを使用
        if hasattr(self, 'operation_period'):
            start_idx = self.operation_period['start_idx']
            end_idx = self.operation_period['end_idx']
            time_data = self.data['Time'].iloc[start_idx:end_idx+1].values
        else:
            time_data = self.data['Time'].values
            
        original_data = self.smoothed_results['original']['data']
        
        # Gaussian手法の比較
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        # Gaussianフィルタ比較
        ax = axes[0, 0]
        ax.plot(time_data, original_data, 'b-', alpha=0.6, linewidth=1, label='Original')
        gaussian_methods = ['gaussian_light', 'gaussian_medium', 'gaussian_strong', 'gaussian_very_strong']
        colors = ['lightcoral', 'red', 'darkred', 'maroon']
        
        for method, color in zip(gaussian_methods, colors):
            if method in self.smoothed_results:
                ax.plot(time_data, self.smoothed_results[method]['data'], 
                       color=color, linewidth=2, alpha=0.8,
                       label=self.smoothed_results[method]['name'])
        
        ax.set_title('Gaussian Filter Comparison')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Isat (A)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Savitzky-Golay比較
        ax = axes[0, 1]
        ax.plot(time_data, original_data, 'b-', alpha=0.6, linewidth=1, label='Original')
        savgol_methods = ['savgol_5_2', 'savgol_7_2', 'savgol_11_2', 'savgol_15_3']
        colors = ['lightgreen', 'green', 'darkgreen', 'forestgreen']
        
        for method, color in zip(savgol_methods, colors):
            if method in self.smoothed_results:
                ax.plot(time_data, self.smoothed_results[method]['data'], 
                       color=color, linewidth=2, alpha=0.8,
                       label=self.smoothed_results[method]['name'])
        
        ax.set_title('Savitzky-Golay Filter Comparison')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Isat (A)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 移動平均比較
        ax = axes[1, 0]
        ax.plot(time_data, original_data, 'b-', alpha=0.6, linewidth=1, label='Original')
        moving_methods = ['moving_avg_3', 'moving_avg_5', 'moving_avg_7', 'moving_avg_11']
        colors = ['lightsalmon', 'orange', 'darkorange', 'orangered']
        
        for method, color in zip(moving_methods, colors):
            if method in self.smoothed_results:
                ax.plot(time_data, self.smoothed_results[method]['data'], 
                       color=color, linewidth=2, alpha=0.8,
                       label=self.smoothed_results[method]['name'])
        
        ax.set_title('Moving Average Comparison')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Isat (A)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 適応的スムージング比較
        ax = axes[1, 1]
        ax.plot(time_data, original_data, 'b-', alpha=0.6, linewidth=1, label='Original')
        adaptive_methods = ['adaptive_light', 'adaptive_medium', 'adaptive_strong']
        colors = ['mediumpurple', 'purple', 'indigo']
        
        for method, color in zip(adaptive_methods, colors):
            if method in self.smoothed_results:
                ax.plot(time_data, self.smoothed_results[method]['data'], 
                       color=color, linewidth=2, alpha=0.8,
                       label=self.smoothed_results[method]['name'])
        
        ax.set_title('Adaptive Smoothing Comparison')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Isat (A)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            filename = 'smoothing_category_comparison.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Category comparison saved as: {filename}")
        
        plt.show()
    
    def _plot_metrics_comparison(self, save_plots):
        """メトリクス比較プロット"""
        metrics = self.calculate_metrics()
        if not metrics:
            return
        
        # メトリクスの可視化
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        methods = list(metrics.keys())
        noise_reduction = [metrics[m]['noise_reduction'] for m in methods]
        correlation = [metrics[m]['correlation'] for m in methods]
        smoothness = [metrics[m]['smoothness'] for m in methods]
        high_value_preservation = [metrics[m]['high_value_preservation'] for m in methods]
        
        # ノイズ削減効果
        ax = axes[0, 0]
        bars = ax.bar(range(len(methods)), noise_reduction, alpha=0.7)
        ax.set_title('Noise Reduction (%)')
        ax.set_ylabel('Noise Reduction (%)')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([metrics[m]['name'] for m in methods], rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # 相関係数
        ax = axes[0, 1]
        bars = ax.bar(range(len(methods)), correlation, alpha=0.7, color='green')
        ax.set_title('Correlation with Original')
        ax.set_ylabel('Correlation')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([metrics[m]['name'] for m in methods], rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # 平滑度
        ax = axes[1, 0]
        bars = ax.bar(range(len(methods)), smoothness, alpha=0.7, color='orange')
        ax.set_title('Smoothness (lower is smoother)')
        ax.set_ylabel('Smoothness')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([metrics[m]['name'] for m in methods], rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # 高値保持率
        ax = axes[1, 1]
        bars = ax.bar(range(len(methods)), high_value_preservation, alpha=0.7, color='purple')
        ax.set_title('High Value Preservation')
        ax.set_ylabel('Preservation')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([metrics[m]['name'] for m in methods], rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            filename = 'smoothing_metrics_comparison.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Metrics comparison saved as: {filename}")
        
        plt.show()
        
        # メトリクス表を表示
        self._print_metrics_table(metrics)
    
    def _print_metrics_table(self, metrics):
        """メトリクステーブルを表示"""
        print("\n" + "="*100)
        print("SMOOTHING METHODS COMPARISON TABLE")
        print("="*100)
        print(f"{'Method':<35} {'Noise Red.':<12} {'Correlation':<12} {'Smoothness':<12} {'High Val. Pres.':<15}")
        print("-"*100)
        
        for method, metric in metrics.items():
            print(f"{metric['name']:<35} {metric['noise_reduction']:<12.2f} "
                  f"{metric['correlation']:<12.4f} {metric['smoothness']:<12.6f} "
                  f"{metric['high_value_preservation']:<15.4f}")
        
        print("="*100)
        
        # 推奨手法を提案
        best_correlation = max(metrics.items(), key=lambda x: x[1]['correlation'])
        best_noise_reduction = max(metrics.items(), key=lambda x: x[1]['noise_reduction'])
        best_balanced = max(metrics.items(), 
                          key=lambda x: x[1]['correlation'] * x[1]['noise_reduction'] / 100 * x[1]['high_value_preservation'])
        
        print("\nRECOMMENDATIONS:")
        print(f"  Best correlation preservation: {best_correlation[1]['name']}")
        print(f"  Best noise reduction: {best_noise_reduction[1]['name']}")
        print(f"  Best balanced approach: {best_balanced[1]['name']}")
    
    def _plot_recommended_methods(self, save_plots):
        """推奨手法の詳細比較"""
        # 運転期間のデータを使用
        if hasattr(self, 'operation_period'):
            start_idx = self.operation_period['start_idx']
            end_idx = self.operation_period['end_idx']
            time_data = self.data['Time'].iloc[start_idx:end_idx+1].values
        else:
            time_data = self.data['Time'].values
            
        original_data = self.smoothed_results['original']['data']
        
        # 推奨手法を選択
        recommended = ['gaussian_medium', 'savgol_7_2', 'adaptive_medium']
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        # 全体比較
        ax = axes[0]
        ax.plot(time_data, original_data, 'b-', alpha=0.6, linewidth=1, label='Original')
        
        colors = ['red', 'green', 'purple']
        for method, color in zip(recommended, colors):
            if method in self.smoothed_results:
                ax.plot(time_data, self.smoothed_results[method]['data'], 
                       color=color, linewidth=2, alpha=0.8,
                       label=self.smoothed_results[method]['name'])
        
        ax.set_title('Recommended Methods Comparison - Full View')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Isat (A)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 拡大表示（興味深い領域）
        ax = axes[1]
        
        # データの中央部分を拡大
        start_idx = len(time_data) // 3
        end_idx = 2 * len(time_data) // 3
        
        ax.plot(time_data[start_idx:end_idx], original_data[start_idx:end_idx], 
               'b-', alpha=0.6, linewidth=1, label='Original')
        
        for method, color in zip(recommended, colors):
            if method in self.smoothed_results:
                ax.plot(time_data[start_idx:end_idx], 
                       self.smoothed_results[method]['data'][start_idx:end_idx], 
                       color=color, linewidth=2, alpha=0.8,
                       label=self.smoothed_results[method]['name'])
        
        ax.set_title('Recommended Methods Comparison - Detailed View')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Isat (A)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            filename = 'smoothing_recommended_comparison.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Recommended methods comparison saved as: {filename}")
        
        plt.show()
    
    def run_comprehensive_analysis(self, target_column='Iis_7L@20'):
        """包括的な分析を実行"""
        print("Starting comprehensive smoothing analysis...")
        print("="*60)
        
        # データ読み込み
        if not self.load_data():
            return False
        
        # データ特性分析
        stats_dict, outlier_indices = self.analyze_data_characteristics(target_column)
        
        # スムージング手法適用
        if not self.apply_smoothing_methods(target_column):
            return False
        
        # 可視化
        self.visualize_all_comparisons()
        
        print("\nAnalysis completed successfully!")
        return True

def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze and compare smoothing methods for raw Isat data')
    parser.add_argument('--data-file', 
                       default='/Users/maesawayuta/yamada_lab/fusion-plasma-ml-toolkit/test/raw_data/DivIis_tor_sum@163402_1.txt',
                       help='Path to the raw data file')
    parser.add_argument('--column', default='Iis_7L@20',
                       help='Target column for analysis (default: Iis_7L@20)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation')
    
    args = parser.parse_args()
    
    # 分析実行
    visualizer = RawDataSmoothingVisualizer(args.data_file)
    success = visualizer.run_comprehensive_analysis(args.column)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
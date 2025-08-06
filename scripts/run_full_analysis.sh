#!/bin/bash

# 核融合プラズマML解析の完全自動実行スクリプト
# Usage: ./scripts/run_full_analysis.sh [YYYYMMDD]

set -e  # エラー時に停止

DATE=${1:-$(date +%Y%m%d)}

echo "=== 核融合プラズマML解析開始 (日付: $DATE) ==="

# 1. データ収集
echo "1. データ収集中..."
cd ../data/makedata
python plasma_data_collector.py
cd ../../

# 2. SVM解析（全パラメータ）
echo "2. SVM解析実行中..."
cd config
make run $DATE
cd ..

# 3. 結果評価
echo "3. F1スコア評価中..."
python src/analysis/f1_score.py -d $DATE

# 4. 結果可視化
echo "4. 結果可視化中..."
python src/analysis/result_plotting.py
python src/preprocessing/separation_region_plotter.py

echo "=== 解析完了 ==="
echo "結果は以下に保存されました:"
echo "- データ: outputs/results/$DATE/"
echo "- プロット: outputs/plots/"
echo "- 処理結果: outputs/process/$DATE/"
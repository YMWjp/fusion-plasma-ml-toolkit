# 🚀 バッチ分析ツール - 複数データファイル一括処理

複数の raw_data ファイルを自動処理し、それぞれの結果を個別のフォルダに整理して格納するバッチ処理ツールです。

## 📁 ファイル構成

### 🔧 バッチ処理ツール

- **`batch_analysis.py`** - メインバッチ処理スクリプト
- **`setup_test_data.py`** - テストデータセットアップ用ユーティリティ

### 📊 入力・出力

- **`raw_data/`** - 生データファイル格納ディレクトリ
- **`results/`** - 分析結果格納ディレクトリ（自動作成）

## 🎯 使用方法

### 1. 📂 データファイルの準備

raw_data ディレクトリに複数のデータファイルを配置します：

```bash
raw_data/
├── DivIis_tor_sum@163402_1.txt
├── DivIis_tor_sum@163403_1.txt
├── DivIis_tor_sum@163404_1.txt
└── other_data_files.txt
```

### 2. 🚀 バッチ分析の実行

#### 基本実行

```bash
python batch_analysis.py
```

#### カスタム設定

```bash
# カスタム入出力ディレクトリ指定
python batch_analysis.py --raw-data-dir my_data --output-dir my_results

# 出力ディレクトリのクリーンアップ付き実行
python batch_analysis.py --clean-output

# 特定センサーのみ分析
python batch_analysis.py --columns Iis_4L@20 Iis_8R@20
```

### 3. 🧪 テストデータの準備

テスト用に複数ファイルを自動生成：

```bash
# テストファイル作成
python setup_test_data.py setup

# テストファイル削除
python setup_test_data.py clean
```

## 📊 出力構造

### 📁 個別ショット結果

各ショットの分析結果は個別のディレクトリに格納されます：

```
results/
├── shot_163402_20250806_195641/
│   ├── all_isat_original_comparison.png      # 全センサー生データ比較
│   ├── all_isat_smoothing_comparison.png     # 全センサースムージング比較
│   ├── top_sensors_recommended_smoothing.png # 上位センサー推奨手法
│   ├── sensor_Iis_4L_at_20/                  # 個別センサー詳細分析
│   │   ├── Iis_4L@20_smoothing_overall_comparison.png
│   │   ├── Iis_4L@20_smoothing_category_comparison.png
│   │   ├── Iis_4L@20_smoothing_metrics_comparison.png
│   │   └── Iis_4L@20_smoothing_recommended_comparison.png
│   ├── sensor_Iis_8R_at_20/
│   └── sensor_Iis_9L_at_17/
├── shot_163403_20250806_195712/
├── shot_163404_20250806_195807/
└── batch_summary.md                          # 全体サマリーレポート
```

### 📋 自動生成レポート

#### 1. **個別ショットサマリー** (`analysis_summary.md`)

- プラズマ運転期間情報
- センサーランキング表
- 推奨事項
- 生成ファイル一覧

#### 2. **バッチ処理サマリー** (`batch_summary.md`)

- 処理成功/失敗統計
- 各ショットの結果一覧
- エラー詳細（該当時）

## 🎯 分析内容

### 🔍 各ショットで実行される分析

1. **🚀 プラズマ運転期間自動検出**

   - 全センサー活動度による期間特定
   - 無意味なゼロ付近データの自動除外

2. **📊 全センサー包括比較**

   - 14 個の Isat センサー一括分析
   - 統計的指標による品質評価
   - 視覚的比較マトリクス

3. **🎯 上位 3 センサー詳細分析**
   - 15 種類のスムージング手法比較
   - 定量的メトリクス評価
   - 推奨手法特定

### 📈 自動評価指標

- **信号強度**: センサーの信号レベル
- **SNR**: 信号対ノイズ比
- **外れ値耐性**: 異常値への耐性
- **ダイナミックレンジ**: 信号の動的範囲
- **ノイズ削減率**: スムージング効果
- **相関保持**: 元信号との類似度

## 🛠️ カスタマイズ

### パラメータ調整

```python
# batch_analysis.py内で調整可能
activity_threshold = 0.05    # 運転期間検出閾値（最大値の5%）
margin_ratio = 0.05          # 運転期間前後マージン（5%）
top_sensors_count = 3        # 詳細分析対象センサー数
```

### ファイル名パターン

支援するファイル名パターン：

- `DivIis_tor_sum@123456_1.txt`
- `data_123456.dat`
- `shot_123456.txt`
- その他の数値パターン

## 🎉 実行例

### 入力例

```bash
$ ls raw_data/
DivIis_tor_sum@163402_1.txt  DivIis_tor_sum@163404_1.txt
DivIis_tor_sum@163403_1.txt  DivIis_tor_sum@163405_1.txt

$ python batch_analysis.py --clean-output
```

### 出力例

```
🚀 Starting batch analysis...
📁 Raw data directory: raw_data
📁 Output base directory: results
Found 4 data files:
  - DivIis_tor_sum@163402_1.txt (447.6 KB)
  - DivIis_tor_sum@163403_1.txt (447.6 KB)
  - DivIis_tor_sum@163404_1.txt (447.6 KB)
  - DivIis_tor_sum@163405_1.txt (447.6 KB)

📊 Processing 4 files...

[1/4] Processing: DivIis_tor_sum@163402_1.txt
============================================================
Analyzing: DivIis_tor_sum@163402_1.txt
============================================================
Detected shot number: 163402
Output directory: results/shot_163402_20250806_195641

Detecting plasma operation period...
  Plasma operation detected:
    Start time: 2.831 s (index: 707)
    End time: 5.839 s (index: 1459)
    Duration: 3.008 s
    Operation ratio: 30.1% of total time

✅ Analysis completed for shot 163402
📁 Results saved in: results/shot_163402_20250806_195641

...

🎉 Batch analysis completed!
✅ Successfully processed: 4 files
❌ Failed: 0 files
```

## 💡 主要利点

### 🚀 **効率性**

- **一括処理**: 複数ファイルの無人実行
- **自動整理**: 結果の体系的分類
- **時系列管理**: タイムスタンプ付きディレクトリ

### 🎯 **精度向上**

- **運転期間自動検出**: 意味のあるデータのみ分析
- **包括評価**: 全センサーの客観的比較
- **最適化推奨**: 最適なセンサー・手法の自動特定

### 📊 **レポート生成**

- **自動ドキュメント**: マークダウン形式サマリー
- **視覚的比較**: 高品質な分析グラフ
- **追跡可能性**: 完全な処理履歴

### 🔧 **拡張性**

- **設定柔軟性**: パラメータの簡単調整
- **フォーマット対応**: 多様なファイル形式
- **カスタマイズ**: 分析手法の追加・変更

この **バッチ分析ツール** により、大量のプラズマ実験データを効率的かつ一貫性をもって処理し、各ショットに最適なセンサーとスムージング手法を自動特定できます！🚀✨

## 🔗 関連ツール

- [`smoothing_test.py`](./README.md#1-個別センサーの詳細分析) - 個別センサー詳細分析
- [`all_isat_comparison.py`](./README.md#2-全センサーの包括的比較) - 全センサー比較分析
- [メイン README](./README.md) - 基本ツールの使用方法

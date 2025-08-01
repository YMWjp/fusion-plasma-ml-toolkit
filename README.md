# 核融合プラズマ機械学習ツールキット

**Fusion Plasma Machine Learning Toolkit**

---

## 概要

本プロジェクトは、核融合プラズマの様々な現象を機械学習で解析することを目的とし、機械学習（SVM、深層学習、回帰分析等）を用いた包括的なデータ分析および可視化を行います。主な処理内容としては、実験データの収集、前処理、SVM による解析、結果の評価（F1 スコア）、および多様なプロットの生成が含まれます。これにより、プラズマ中の様々な物理現象の理解と予測を行います。

---

## ディレクトリ構成

```
fusion-plasma-ml-toolkit/
├── config/
│   └── Makefile                   # 一括実行用Makefile
├── scripts/
│   └── run_full_analysis.sh       # 完全自動化スクリプト
├── data/
│   ├── makedata/                  # データ収集・前処理
│   │   ├── plasma_data_collector.py     # データ収集・CSV出力
│   │   ├── shot_data_visualizer.py      # データ可視化
│   │   ├── hysteresis_plotter.py        # ヒステリシスプロット
│   │   ├── classes/              # データ処理クラス
│   │   └── get_params/           # パラメータ取得
│   ├── processed/                # 処理済みデータ
│   └── raw/                      # 生データ
├── src/
│   ├── analysis/                 # データ解析
│   │   ├── f1_score.py          # F1スコア評価
│   │   ├── result_plotting.py   # 結果プロット
│   │   └── svm_analysis.py      # SVM解析
│   ├── preprocessing/            # 前処理
│   │   ├── svm_exhaustive_search.py     # SVM最適化
│   │   └── separation_region_plotter.py # 分離領域図作成
│   ├── detection/               # 変化点検出
│   │   └── change_detection.py
│   └── utils/                   # ユーティリティ
│       └── common.py
├── outputs/                     # 出力結果
│   ├── plots/                  # プロット画像
│   ├── process/               # 処理結果
│   └── results/              # 最終結果
├── docs/                     # ドキュメント
│   ├── README_JP.md         # 日本語詳細ドキュメント
│   └── README_EN.md         # 英語版ドキュメント
└── requirements.txt         # 依存パッケージ
```

---

## 必要なパッケージ

本プロジェクトを実行するために必要な Python パッケージは以下の通りです。`requirements.txt`を使用して一括インストールすることを推奨します。

```bash
pip install -r requirements.txt
```

主なパッケージ：

- numpy
- matplotlib
- pandas
- scikit-learn
- docopt
- japanize_matplotlib
- scipy
- joblib

---

## クイックスタート

### 🚀 完全自動化実行（推奨）

```bash
# 環境構築
pip install -r requirements.txt

# 全処理を自動実行
./scripts/run_full_analysis.sh [YYYYMMDD]
```

### 📋 手動実行

#### 1. 環境構築

```bash
pip install -r requirements.txt
```

#### 2. データ収集

```bash
cd data/makedata
python plasma_data_collector.py
```

#### 3. SVM 解析の実行

```bash
# 個別実行
python src/preprocessing/svm_exhaustive_search.py [YYYYMMDD] [K(1~14)]

# 一括実行（推奨）
cd config
make run [YYYYMMDD]
```

#### 4. 結果の可視化と評価

```bash
# F1スコア評価
python src/analysis/f1_score.py -d [YYYYMMDD]

# 結果プロット
python src/analysis/result_plotting.py

# 分離領域図作成
python src/preprocessing/separation_region_plotter.py
```

---

## 主要スクリプトの説明

### データ収集・前処理

| ファイル                                 | 説明                           | 使用方法                          |
| ---------------------------------------- | ------------------------------ | --------------------------------- |
| `data/makedata/plasma_data_collector.py` | 実験データの収集と CSV 出力    | `python plasma_data_collector.py` |
| `data/makedata/shot_data_visualizer.py`  | 特定ショット番号のデータ可視化 | `python shot_data_visualizer.py`  |
| `data/makedata/hysteresis_plotter.py`    | ヒステリシスプロット生成       | `python hysteresis_plotter.py`    |

### SVM 解析

| ファイル                                     | 説明                                | 使用方法                                     |
| -------------------------------------------- | ----------------------------------- | -------------------------------------------- |
| `src/preprocessing/svm_exhaustive_search.py` | Exhaustive Search による SVM 最適化 | `python svm_exhaustive_search.py [date] [K]` |
| `src/analysis/svm_analysis.py`               | SVM 解析実行                        | `python svm_analysis.py [date] [seed]`       |

### 結果評価・可視化

| ファイル                                         | 説明                       | 使用方法                              |
| ------------------------------------------------ | -------------------------- | ------------------------------------- |
| `src/analysis/f1_score.py`                       | F1 スコアによる評価        | `python f1_score.py [options]`        |
| `src/analysis/result_plotting.py`                | 結果の散布図プロット       | `python result_plotting.py`           |
| `src/preprocessing/separation_region_plotter.py` | 1 パラメータ分離領域図作成 | `python separation_region_plotter.py` |

### その他

| ファイル                            | 説明                 | 使用方法                     |
| ----------------------------------- | -------------------- | ---------------------------- |
| `src/detection/change_detection.py` | 変化点検出・異常検知 | `python change_detection.py` |

---

## 実行手順（詳細）

### ステップ 1: データ収集

```bash
cd data/makedata
python plasma_data_collector.py
```

- 指定されたショット番号のデータをサーバから収集
- CSV 形式で保存

### ステップ 2: データ可視化（オプション）

```bash
python shot_data_visualizer.py  # 個別ショット可視化
python hysteresis_plotter.py    # ヒステリシス解析
```

### ステップ 3: SVM 解析

```bash
# 方法1: 個別実行
python src/preprocessing/svm_exhaustive_search.py 20240923 1

# 方法2: 一括実行（推奨）
cd config
make run 20240923
```

### ステップ 4: 結果評価

```bash
# F1スコア計算
python src/analysis/f1_score.py -d 20240923

# 結果可視化
python src/analysis/result_plotting.py

# 分離領域図
python src/preprocessing/separation_region_plotter.py
```

### ステップ 5: 変化点検出（オプション）

```bash
python src/detection/change_detection.py
```

---

## 注意事項

- 各スクリプトの実行前に、必要なデータファイルが`./outputs/results/[DATE]/`ディレクトリに存在することを確認してください
- 日付形式は`YYYYMMDD`形式で指定してください（例：20240923）
- Makefile を使用した一括実行を推奨します
- パラメータ K は 1~14 の範囲で指定してください

---

## トラブルシューティング

### よくある問題

1. **ImportError**: `pip install -r requirements.txt`で依存関係を再インストール
2. **ファイルが見つからない**: データファイルのパスと存在を確認
3. **メモリエラー**: より小さなデータセットで実行を試行

### サポート

問題が発生した場合は、以下にお問い合わせください：

- Email: maesawa-yuta436@g.ecc.u-tokyo.ac.jp

---

## 作成者

- **名前**: 前澤 佑太
- **所属**: 東京大学工学部システム創成学科 E&E コース 山田研究室
- **Email**: maesawa-yuta436@g.ecc.u-tokyo.ac.jp
- **LinkedIn**: [前澤 佑太](https://www.linkedin.com/in/yuta-maesawa/)

---

## ライセンス

本プロジェクトに関する詳細な使用条件については、作成者にお問い合わせください。

---

## 貢献

本プロジェクトへの貢献は歓迎します。バグ報告や機能提案、プルリクエストなどをご自由にお寄せください。

**English documentation is available in `docs/README_EN.md`**

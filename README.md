# 核融合プラズマ機械学習ツールキット

**Fusion Plasma Machine Learning Toolkit**

---

## 概要

本プロジェクトは、核融合プラズマの様々な現象を機械学習で解析することを目的とし、機械学習（SVM、深層学習、回帰分析等）を用いた包括的なデータ分析および可視化を行います。主な処理内容としては、実験データの収集、前処理、SVM による解析、結果の評価（F1 スコア）、および多様なプロットの生成が含まれます。これにより、プラズマ中の様々な物理現象の理解と予測を行います。

---

## ディレクトリ構成

```
fusion-plasma-ml-toolkit/
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

```

### 📋 手動実行

#### 1. 環境構築

```bash
pip install -r requirements.txt
```

#### 2. データ収集

```bash
python -m src.main
```

#### 3. SVM 解析の実行

```bash
# 個別実行
未実装
```

#### 4. 結果の可視化と評価

```bash
未実装
```

---

### サポート

問題が発生した場合は、以下にお問い合わせください：

- Email: maesawa-yuta436@g.ecc.u-tokyo.ac.jp

---

## 作成者

- **名前**: 前澤 佑太
- **所属**: 東京大学工学部システム創成学科 E&E コース 齋藤研究室
- **Email**: maesawa-yuta436@g.ecc.u-tokyo.ac.jp
- **LinkedIn**: [前澤 佑太](https://www.linkedin.com/in/yuta-maesawa/)

---

## ライセンス

本プロジェクトに関する詳細な使用条件については、作成者にお問い合わせください。

---

## 貢献

本プロジェクトへの貢献は歓迎します。バグ報告や機能提案、プルリクエストなどをご自由にお寄せください。

**English documentation is available in `docs/README_EN.md`**

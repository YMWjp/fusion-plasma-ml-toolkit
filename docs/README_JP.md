# プロジェクト名

**Radiation-Collapse Study with SVM Analysis**

---

## 概要

本プロジェクトは、放射崩壊現象の研究を目的とし、Support Vector Machine（SVM）を用いたデータ分析および可視化を行います。主な処理内容としては、実験データの収集、前処理、SVM による解析、結果の評価（F1 スコア）、および多様なプロットの生成が含まれます。これにより、放電現象における重要なパラメータ間の相関関係や異常検知を行います。

---

## ディレクトリ構成

```
project_root/
├── makedata/
│   ├── pr7_25.py
│   ├── makeimage.py
│   ├── makehysteresis.py
│   ├── classes/
│   │   └── CalcMPEXP.py
│   └── ...
├── classes/
│   └── ...
├── scripts/
│   ├── svm_result_analysis_and_plot.py
│   ├── pr8.py
│   ├── pr9_5.py
│   ├── change_detection.py
│   ├── F1score.py
│   └── ES_SVM.py
├── common/
│   └── names_dict.py
├── results/
│   └── ...
├── hist&sccaterpng/
│   └── ...
├── README.md
└── requirements.txt
```

---

## 必要なパッケージ

本プロジェクトを実行するために必要な Python パッケージは以下の通りです。`requirements.txt`を使用して一括インストールすることを推奨します。

```bash
pip install -r requirements.txt
```

以下は主なパッケージの一覧です：

- numpy
- matplotlib
- pandas
- scikit-learn
- docopt
- japanize_matplotlib
- scipy
- joblib

---

## データ収集・前処理

### `makedata/pr7_25.py`

データ収集および前処理を行うスクリプトです。特定のショット番号に対して必要な物理パラメータをサーバから取得し、CSV ファイルとして保存します。

**使用方法:**

```bash
cd makedata
python pr7_25.py
```

### `makedata/makeimage.py`

収集したデータを用いて、特定のショット番号（例：115083）の各パラメータを可視化します。複数のグラフを生成し、結果を画像ファイルとして保存します。

**使用方法:**

```bash
cd makedata
python makeimage.py
```

### `makedata/makehysteresis.py`

選択されたショット番号に対して、時系列データの外れ値検出およびスムージングを行い、ヒステリシスプロットを生成します。

**使用方法:**

```bash
cd makedata
python makehysteresis.py
```

---

## データ解析

### `svm_result_analysis_and_plot.py`

SVM の結果を解析し、散布図をプロットするメインスクリプトです。データの読み込み、重みとバイアスの計算、関数評価、プロットの生成を行います。

**使用方法:**

```bash
python svm_result_analysis_and_plot.py
```

### `pr8.py`

Exhaustive Search（総当たり検索）を用いて SVM の最適なパラメータを探索するスクリプトです。指定されたデータセットとパラメータリストに基づいて解析を実行します。

**使用方法:**

```bash
python src/preprocessing/pr8.py [date] [K(1~14)]
```

または

```bash
make run [data]
```

一括実行も可能です。

### `ES_SVM.py`

Exhaustive Search を実装したクラス`ExhaustiveSearch`を含むモジュールです。SVM モデルのパラメータを総当たりで探索し、交差検証を通じて最適なモデルを選定します。

**使用方法:**

```bash
python src/analysis/svm_analysis.py [date] [seed]
```

### `F1score.py`

SVM の解析結果を F1 スコアで評価し、最適なモデルの選定を補助するスクリプトです。異なるパラメータ組み合わせの F1 スコアを比較します。

**使用方法:**

```bash
python src/analysis/f1_score.py [オプション]
```

オプションには、DoS 図の描画、複数シードの処理、特定プロジェクト用の設定などがあります。詳細はスクリプト内のドキュメントを参照してください。

### `change_detection.py`

選択されたショット番号に対して、データの変化点検出を行い、異常なデータポイントを特定・可視化します。

**使用方法:**

```bash
python change_detection.py
```

---

## データ管理クラス

### `makedata/classes/CalcMPEXP.py`

データの計算およびプロットを行うクラス`CalcMPEXP`を含むモジュールです。取得したデータを基に、各種物理パラメータの計算や可視化をサポートします。

---

## その他の重要ファイル

### `common/names_dict.py`

プロジェクト全体で使用されるパラメータ名とその説明を格納する辞書を定義しています。各スクリプトからインポートして使用します。

### `README.md`

本ドキュメントです。プロジェクトの概要、使用方法、ディレクトリ構成などを説明しています。

### `requirements.txt`

プロジェクトで必要となる Python パッケージの一覧を記載しています。

---

## 実行手順

1. **環境構築**

   必要なパッケージをインストールします。

   ```bash
   pip install -r requirements.txt
   ```

2. **データ収集**

   ショット番号に基づいたデータを収集し、CSV ファイルとして保存します。

   ```bash
   cd makedata
   python pr7_25.py
   ```

3. **データ可視化**

   収集したデータを用いて画像を生成します。

   ```bash
   python makeimage.py
   python makehysteresis.py
   ```

4. **SVM 解析**

   Exhaustive Search を実行し、SVM モデルの最適パラメータを探索します。

   ```bash
   python src/preprocessing/pr8.py [date] [K(1~14)]
   ```

   または

   ```bash
   make run [data]
   ```

5. **結果解析とプロット**

   SVM の結果を解析し、散布図を生成します。

   ```bash
   python svm_result_analysis_and_plot.py
   ```

6. **評価**

   F1 スコアを用いて解析結果を評価します。

   ```bash
   python src/analysis/f1_score.py [オプション]
   ```

7. **変化点検出**

   データの異常検知を実行します。

   ```bash
   python change_detection.py
   ```

---

## 注意事項

- 各スクリプトの実行前に、必要なデータファイルが`./results/[DATE]/`ディレクトリに存在することを確認してください。

- `F1score.py`や`pr8.py`などのスクリプトでは、実行時に必要な引数を適切に指定してください。詳細な使用方法は各スクリプト内のドキュメントやコメントを参照してください。

- データの前処理や解析結果の精度向上のため、スクリプト内のパラメータ設定を必要に応じて調整してください。

---

## 制作者プロフィール

- **名前**: 前澤 佑太
- **所属**: 東京大学工学部システム創成学科 E&E コース 山田研究室
- **メールアドレス**: maesawa-yuta436@g.ecc.u-tokyo.ac.jp
- **LinkedIn**: [前澤 佑太の LinkedIn](https://www.linkedin.com/in/yuta-maesawa/)

---

## 参考文献

- 整理中

---

## 更新履歴(現在製作中)

- **20XX-00-00**: 初版作成

---

## 貢献方法

本プロジェクトへの貢献は歓迎します。バグ報告や機能提案、プルリクエストなどをご自由にお寄せください。

---

# おわりに

本プロジェクトがプラズマの接触、及び非接触状態の理解と解析に役立つことを願っています。皆様のご協力とフィードバックをお待ちしております。

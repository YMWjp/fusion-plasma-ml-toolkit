# プロジェクト構成と責務分離ガイド

## ディレクトリ構成（推奨）

```
src/
├─ main.py                         # エントリポイント
├─ config/                         # 設定ファイル読み込み
├─ utils/                          # 小規模共通関数とパス定義
├─ domain/                         # 核となる業務/研究ロジック（I/O依存なし）
├─ application/                    # ユースケース実装（層の調停）
├─ infrastructure/                 # 外部I/O（CSV, DB, API）
└─ data/
   ├─ datasets/                    # 出力CSV
   ├─ logs/                        # ログファイル
   ├─ egdata/
   ├─ SDLloopdata/
   └─ experiment_log_new.csv
```

## 各層の責務

- **main.py**: ユースケース呼び出しのみ
- **config**: 設定ファイル読込
- **utils**: どの層にも依存しない小道具（ヘッダ書き込み・ログ）
- **domain**: 純粋なロジック（計算・ルール）
- **infrastructure**: ファイル/外部システムからデータ取得
- **application**: 各層を組み合わせた処理フロー

## 層をまたぐ処理の流れ（例：1 ショット処理）

1. **application**: ユースケース開始
2. **infrastructure**: データ取得（CSV, egdata 等）
3. **domain**: 計算・補間・ラベリングなどのロジック処理
4. **application**: 出力整形
5. **utils**: ファイル保存・ログ出力

## 移行手順（段階的）

1. `paths.py` 完成
2. `utils` に共通 I/O 関数作成
3. infrastructure に repository 作成
4. domain に計算ロジック移植
5. application にユースケース追加
6. 順次、既存スクリプトから移植

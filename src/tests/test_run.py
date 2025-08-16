# test_run.py
from src.domain.params.context import Context
from src.domain.params.executor import compute, list_required_files

# Context はとりあえず最低限
ctx = Context(
    shotNO=12345,
    data_root="/tmp",
    data_sources={"fircall": "{root}/fircall@{shotNO}.txt"}
)

targets = ["nelgrad"]

# 必要ファイル一覧を確認
print("Required files:", list_required_files(ctx, targets))

# 計算を実行
results = compute(ctx, targets)
print("Results:", results)